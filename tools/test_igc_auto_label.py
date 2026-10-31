"""Auto-label on small written traces; no model or GPU."""
import unittest
from tools.igc.auto_label import label_text, summarize


class AutoLabel(unittest.TestCase):
    def one(self, prompt, thinking):
        rows = label_text(prompt, thinking)
        return [r for r in rows if r['label'] == 'computed']

    def test_syntax_is_level_zero(self):
        rows = self.one('Add them.', '68824 * 42716 = 2939885984 is the product')
        self.assertEqual([(r['level'], r['op'], sorted(r['operands'])) for r in rows], [(0, 'mul', ['42716', '68824'])])

    def test_words_in_one_sentence_are_level_one(self):
        rows = self.one('Janet has 180 eggs.', 'She keeps half of them, so 180 divided by 2 gives 90 eggs.')
        self.assertEqual([(r['level'], r['op']) for r in rows], [(1, 'div')])

    def test_earlier_value_is_level_two(self):
        rows = self.one('A box holds 48 pens. Each case has 3 boxes.',
                        'Start with the pens.\nThat was the first line.\nA case then holds 144 pens.')
        self.assertEqual([(r['level'], r['op']) for r in rows], [(2, 'mul')])

    def test_two_explanations_are_dropped(self):
        rows = label_text('We have 6 and 7 and 40 and 2.', 'So we get 42 now.')
        self.assertEqual([r['label'] for r in rows if r['value'] == '42'], ['ambiguous'])

    def test_unrelated_number_is_negative(self):
        rows = label_text('We have 7 and 11.', 'In the year 1999 this was known.')
        self.assertEqual([r['label'] for r in rows if r['value'] == '1999'], ['not_computed'])

    def test_copied_number_is_not_a_result(self):
        rows = label_text('Tom has 15 apples.', 'Tom has 15 apples, as stated.')
        self.assertEqual([r['label'] for r in rows if r['value'] == '15'], ['copied'])

    def test_angles_are_not_results(self):
        rows = label_text('Angles 111.8 and 16.1.', 'The angle is 95.7' + chr(176) + ' here')
        self.assertEqual([r['label'] for r in rows if r['value'] == '95.7'], ['not_computed'])

    def test_exact_decimals(self):
        rows = self.one('Price is 0.1 and tax is 0.2.', 'Total 0.1 + 0.2 = 0.3 here')
        self.assertEqual([r['op'] for r in rows], ['add'])

    def test_summary_counts_levels(self):
        rows = label_text('Add.', '12 + 30 = 42 then 42 / 2 = 21')
        counts = summarize([rows])
        self.assertEqual(counts['computed_level_0'], 2)


if __name__ == '__main__':
    unittest.main()
