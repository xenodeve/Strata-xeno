"""Matched-intent data and final-activation routing are separate from operands."""
import unittest
from collections import Counter
import numpy as np
from tools.igc.readiness_corpus import dataset, anchor_examples


class BytesTokenizer:
    def encode(self,text):return list(text.encode('utf-8'))


class ReadinessCorpus(unittest.TestCase):
    def test_matched_pairs_split_as_a_unit_without_answer_targets(self):
        data=dataset();rows=data['rows']
        self.assertEqual(len(rows),320)
        self.assertEqual(Counter(r['split'] for r in rows),dict(train=208,development=112))
        pairs={}
        for r in rows:
            self.assertNotIn('answer',r);self.assertNotIn('completion',r)
            if r['pair'] is not None:pairs.setdefault(r['pair'],[]).append(r)
        self.assertEqual(len(pairs),144)
        seen=set()
        for pair in pairs.values():
            self.assertEqual(len(pair),2)
            self.assertEqual({r['ready'] for r in pair},{0,1})
            self.assertEqual(len({(r['lhs'],r['rhs'],r['operation'],r['expression'],r['split']) for r in pair}),1)
            self.assertNotIn((pair[0]['lhs'],pair[0]['rhs']),seen)
            seen.add((pair[0]['lhs'],pair[0]['rhs']))
        fit={r['template'] for r in rows if r['split']=='train'}
        dev={r['template'] for r in rows if r['split']=='development'}
        self.assertFalse(fit&dev)
        self.assertEqual(dataset(),data)

    def test_anchor_features_end_before_any_generated_token(self):
        rows=[dict(id='x',split='train',ready=1,prompt='Q')]
        example=anchor_examples(rows,BytesTokenizer(),b'S',b'A')[0]
        self.assertEqual(example['ids'],[83,81,65])
        self.assertIsNone(example['target']);self.assertFalse(example['numeric_target'])
        self.assertEqual(example['position'],2)
        self.assertEqual(example['invocation_position'],2)
        with self.assertRaises(ValueError):anchor_examples(rows*2,BytesTokenizer(),b'S',b'A')


if __name__=='__main__':unittest.main()
