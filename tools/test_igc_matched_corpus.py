"""Matched development data must isolate template/intent from operand changes."""
import collections
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.igc.matched_corpus import build, write_new
from tools.igc.readout import targets, metrics


def pool():
    # Ordinary and critical values in every length/sign stratum; no saved corpus.
    values = list(range(-19, 20))
    return [(a, b) for a in values for b in values]


class MatchedCorpus(unittest.TestCase):
    def test_same_operands_and_expression_across_template_and_intent_conditions(self):
        value = build(pool())
        groups = collections.defaultdict(list)
        for row in value['rows']:
            groups[row['group']].append(row)
        self.assertTrue(groups)
        for rows in groups.values():
            self.assertEqual(len({(r['lhs'], r['rhs'], r['operation'], r['expression']) for r in rows}), 1)
            families = {'seen'} if rows[0]['split'] == 'train' else {'seen', 'new'}
            self.assertEqual({(r['family'], r['language'], r['intent']) for r in rows},
                             {(f, l, i) for f in families for l in ('en', 'th') for i in ('compute', 'quote-copy')})
            for row in rows:
                self.assertEqual(row['ready_op'], row['operation'] if row['intent'] == 'compute' else 0)
                self.assertEqual(row['prompt'].count(row['expression']), 1)

    def test_pair_disjoint_splits_and_equal_length_sign_operation_coverage(self):
        value = build(pool())
        sets = {}
        for split in ('train', 'development'):
            rows = [r for r in value['rows'] if r['split'] == split]
            sets[split] = {(r['lhs'], r['rhs']) for r in rows}
            positive = [r for r in rows if r['intent'] == 'compute' and r['family'] == 'seen' and r['language'] == 'en']
            counts = collections.Counter((len(str(abs(r['lhs']))), len(str(abs(r['rhs']))),
                                          r['lhs'] < 0, r['rhs'] < 0, r['operation']) for r in positive)
            self.assertEqual(len(counts), 48)
            self.assertEqual(set(counts.values()), {4 if split == 'train' else 2})
            for side in ('lhs', 'rhs'):
                self.assertTrue({-8, -3, 3, 8} <= {r[side] for r in rows})
        self.assertFalse(sets['train'] & sets['development'])
        self.assertEqual(len(value['rows']), 1536)

    def test_actual_readout_keeps_padding_errors_wrong(self):
        row = dict(lhs=-3, rhs=8, ready_op=3)
        truth = targets([row])
        self.assertEqual(truth.tolist(), [[3, 1, 0, 0, 0, 3, 0, 8, 0]])
        predicted = truth.copy()
        predicted[0, 6] = 3
        self.assertEqual(metrics(predicted, [row])['positive_exact_routed'], 0)
        self.assertEqual(predicted[0, 6], 3)  # Scoring must not repair the prediction.

    def test_deterministic_bounded_refusal_and_non_overwrite(self):
        self.assertEqual(build(pool()), build(reversed(pool())))
        with self.assertRaises(ValueError):
            build([(0, 0)])
        with self.assertRaises(ValueError):
            build(pool() + [(100, 1)])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'new'
            value = build(pool())
            write_new(output, value)
            before = (output / 'corpus.json').read_bytes()
            with self.assertRaises(FileExistsError):
                write_new(output, value)
            self.assertEqual((output / 'corpus.json').read_bytes(), before)

    def test_cli_refuses_unrecognized_pool_without_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'unrecognized.json'
            source.write_text('{"rows": []}', encoding='utf-8')
            output = root / 'new'
            result = subprocess.run([sys.executable, '-B', '-m', 'tools.igc.matched_corpus',
                '--source-corpus', str(source), '--output', str(output)],
                capture_output=True, text=True, timeout=10,
                creationflags=0x08000000 if sys.platform == 'win32' else 0)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
