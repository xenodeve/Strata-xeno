"""Audit counts must keep incorrect encodings and overlapping causes visible."""
import unittest
from tools.igc.development_audit import summarize, replay_counterfactual, predicted_padding


class DevelopmentAudit(unittest.TestCase):
    def test_padding_variant_uses_predicted_length_and_never_repairs_active_digits(self):
        raw = [1, 1, 0, 1, 0, 0, 3, 8, 8]
        self.assertEqual(predicted_padding(raw), [1, 1, 0, 1, 0, 0, 3, 8, 0])
        self.assertEqual(raw[-1], 8)
        with self.assertRaises(ValueError):
            predicted_padding([1, 0, 0, 2, 0, 3, 0, 8, 0])

    def test_confusion_counts_active_and_padding_without_repair(self):
        rows = [dict(id='a', split='development', ready_op=3, lhs=-3, rhs=8)]
        predictions = [dict(id='a', target=[3, 1, 0, 0, 0, 3, 0, 8, 0],
                            predicted=[3, 1, 0, 0, 0, 4, 3, 8, 8])]
        result = summarize(rows, predictions)['development']
        self.assertEqual(result['exact'], 0)
        self.assertEqual(result['digit_confusion']['lhs/1/ones'], {'3->4': 1})
        self.assertEqual(result['digit_confusion']['lhs/1/tens'], {'0->3': 1})
        self.assertEqual(result['digit_confusion']['rhs/1/tens'], {'0->8': 1})
        self.assertEqual(result.get('error_associations'), dict(ones_errors=1, padding_errors=2,
            ones_matches_other_active_digit=0, padding_matches_own_ones=2))
        self.assertEqual(predictions[0]['predicted'][6], 3)

    def test_counterfactual_keeps_readiness_even_when_both_tuples_are_correct(self):
        truth = [1, 0, 0, 0, 0, 3, 0, 8, 0]
        records = [dict(ready=0, selection=1, primary=truth, secondary=truth,
                        original_label=dict(ready_op=1, lhs=3, rhs=8)),
                   dict(ready=1, selection=4, primary=[1, 0, 0, 0, 0, 4, 0, 8, 0],
                        secondary=truth, original_label=dict(ready_op=1, lhs=3, rhs=8))]
        result = replay_counterfactual(records)
        self.assertEqual(result['positive_selected'], 1)
        self.assertEqual(result['positive_no_call'], 1)
        self.assertEqual(result['rescued_disagreements'], 1)
        self.assertEqual(result['selected_wrong_tuple'], 0)


if __name__ == '__main__':
    unittest.main()
