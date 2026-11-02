"""The full test plan's row builders and the one rule every suite is scored by. No real questions: GPQA's authors ask that theirs are not
shared, so the csv here is made up."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from tools.igc.select_rows import right, verdicts
from tools.igc.standard_rows import LETTER_TAIL, PLAN, build_plan, gpqa_rows, pick

COLUMNS = ['Question', 'Correct Answer', 'Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3', 'Record ID', 'High-level domain']


def write_gpqa(path, rows):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, COLUMNS)
        w.writeheader()
        for i, domain in enumerate(rows):
            w.writerow({'Question': f'q{i}?', 'Correct Answer': f'right{i}', 'Incorrect Answer 1': f'w{i}a', 'Incorrect Answer 2': f'w{i}b',
                        'Incorrect Answer 3': f'w{i}c', 'Record ID': f'rec{i:05d}', 'High-level domain': domain})


class Gpqa(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / 'gpqa_diamond.csv'
        write_gpqa(self.path, ['Physics', 'Chemistry', 'Biology'] * 4)

    def tearDown(self):
        self.dir.cleanup()

    def test_biology_is_left_out(self):
        rows = list(gpqa_rows(self.path))
        self.assertEqual(len(rows), 8)
        self.assertEqual({r['group'] for r in rows}, {'Physics', 'Chemistry'})

    def test_the_letter_points_at_the_right_option(self):
        for r in gpqa_rows(self.path):
            n = r['id'].removeprefix('gpqa-rec').lstrip('0') or '0'
            line = next(x for x in r['prompt'].splitlines() if x.startswith(r['answer'] + ') '))
            self.assertEqual(line, f"{r['answer']}) right{n}")
            self.assertTrue(r['prompt'].endswith(LETTER_TAIL))

    def test_the_same_letters_every_time(self):
        self.assertEqual(list(gpqa_rows(self.path)), list(gpqa_rows(self.path)))

    def test_the_correct_letter_is_not_always_the_same(self):
        self.assertGreater(len({r['answer'] for r in gpqa_rows(self.path)}), 1)


class Pick(unittest.TestCase):
    def test_seeded_and_filtered(self):
        rows = [dict(id=i, level='Level 5' if i % 2 else 'Level 1') for i in range(40)]
        a, passed = pick(rows, 5, PLAN['math5'][2])
        self.assertEqual(passed, 20)
        self.assertEqual(a, pick(rows, 5, PLAN['math5'][2])[0])
        self.assertTrue(all(r['level'] == 'Level 5' for r in a))

    def test_super_gpqa_keeps_calculation_in_math_physics_engineering(self):
        keep = PLAN['supergpqa'][2]
        self.assertTrue(keep(dict(calculation=True, field='Physics', group='Science')))
        self.assertTrue(keep(dict(calculation=True, field='Mechanics', group='Engineering')))
        self.assertFalse(keep(dict(calculation=True, field='Biology', group='Science')))
        self.assertFalse(keep(dict(calculation=False, field='Mathematics', group='Science')))


class Rule(unittest.TestCase):
    def test_a_letter_suite_reads_the_letter(self):
        row = dict(suite='gpqa', answer='C')
        self.assertTrue(right(row, 'x</think>So the answer is (C)'))
        self.assertFalse(right(row, 'x</think>So the answer is (B), not 12'))

    def test_gsm8k_reads_the_number_not_a_letter(self):
        row = dict(suite='gsm8k', answer=42)
        self.assertTrue(right(row, 'x</think>The answer is **42**'))
        self.assertFalse(right(row, 'x</think>The answer is **41**'))
        self.assertFalse(right(row, 'thinking that never closed 42'))

    def test_verdicts_for_stop_on_loss(self):
        records = [dict(id='a', suite='gsm8k', answer=3, text='x</think>\\boxed{3}'), dict(id='b', suite='mmlupro', answer='A', text='x</think>the answer is (B)')]
        self.assertEqual(verdicts(records), {'a': 'ok', 'b': 'wrong'})


class Plan(unittest.TestCase):
    def test_builds_the_stages_whose_source_is_there(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            write_gpqa(d / 'gpqa_diamond.csv', ['Physics'] * 70 + ['Biology'] * 5)
            (d / 'gsm8k-test.jsonl').write_text(''.join(json.dumps(dict(question=f'q{i}', answer=f'work\n#### {i},000')) + '\n' for i in range(100)), encoding='utf-8')
            out = d / 'out'
            out.mkdir()
            build_plan(d, out)
            self.assertEqual(sorted(p.name for p in out.iterdir()), ['std-gpqa60.jsonl', 'std-gsm60.jsonl'])
            gsm = [json.loads(line) for line in (out / 'std-gsm60.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(gsm), 60)
            self.assertTrue(all(r['answer'] == int(r['prompt'][1:]) * 1000 for r in gsm))
            self.assertEqual(len((out / 'std-gpqa60.jsonl').read_text(encoding='utf-8').splitlines()), 60)


if __name__ == '__main__':
    unittest.main()
