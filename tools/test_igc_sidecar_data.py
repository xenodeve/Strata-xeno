"""Sidecar examples from annotated text; no model or GPU."""
import unittest
from tools.igc.sidecar_data import clean, examples
from tools.igc.auto_label import after_expression
from tools.igc.sidecar_traces import rows_of, without_expressions


class SidecarData(unittest.TestCase):
    def test_annotation_leaves_text_and_offset(self):
        text, calls = clean('She sold 48/2 = <<48/2=24>>24 clips.')
        self.assertEqual(text, 'She sold 48/2 = 24 clips.')
        self.assertEqual(calls, [(16, '48/2', '24')])

    def test_pointers_reach_written_numbers(self):
        row = dict(question='Natalia sold 48 clips and half as many in May, at 2 each.',
                   answer='In May she sold 48/2 = <<48/2=24>>24 clips.\n#### 24')
        example = next(examples(row))
        self.assertTrue(example['kept'])
        self.assertEqual((example['op'], example['result']), ('div', '24'))
        self.assertEqual(example['entries'], 4)
        self.assertEqual((example['left'], example['right']), (2, 3))

    def test_number_never_written_is_dropped(self):
        row = dict(question='Tom has 5 pens.', answer='Twice that is <<5*2=10>>10.\n#### 10')
        self.assertEqual(next(examples(row)), dict(kept=False, reason='operand not written'))

    def test_copied_number_is_a_negative_and_computed_a_positive(self):
        rows = list(rows_of('Tom has 37 pens and 41 bags.', 'He has 37 pens, so the total comes to 78 items.'))
        self.assertEqual([r['op'] for r in rows], ['none', 'add'])
        self.assertEqual((rows[1]['left'], rows[1]['right']), (2, 1))

    def test_small_result_is_not_taught_as_no_calculation(self):
        rows = list(rows_of('A wire is 48 inches, cut into pieces 6 long.', 'That makes 8 pieces.'))
        self.assertEqual(rows, [])

    def test_parser_slots_leave_the_sidecar(self):
        self.assertTrue(after_expression('Total 20+80+160='))
        self.assertTrue(after_expression('cost 3 * $4 = $'))
        self.assertFalse(after_expression('Total = '))
        self.assertFalse(after_expression('half of 180 is '))
        rows = list(rows_of('Tom has 37 pens and 41 bags.', 'So 37+41=78 items.'))
        self.assertEqual([r['text'][r['slot']:r['slot'] + 2] for r in rows], ['37', '41'])

    def test_expressions_leave_plain_words(self):
        self.assertEqual(without_expressions('So 37+41=78 items, then 3 * $4 = $12 left.'), 'So 78 items, then 12 left.')
        self.assertEqual(without_expressions('Total 20+80+160=260.'), 'Total 260.')
        self.assertEqual(without_expressions('x = 5 and 12 items.'), 'x = 5 and 12 items.')

    def test_two_operations_are_dropped(self):
        row = dict(question='Tom has 5 pens and 2 bags.', answer='So <<5*2+1=11>>11.\n#### 11')
        self.assertEqual(next(examples(row)), dict(kept=False, reason='not one operation'))


if __name__ == '__main__':
    unittest.main()
