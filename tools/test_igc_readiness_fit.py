import unittest
import numpy as np
from tools.igc.readiness_fit import metrics


class ReadinessScoring(unittest.TestCase):
    def test_false_call_does_not_pass_a_matched_pair(self):
        rows=[dict(id='a',pair='p',ready=1,template='yes',language='en'),
              dict(id='b',pair='p',ready=0,template='no',language='en'),
              dict(id='c',pair=None,ready=0,template='ordinary',language='th')]
        perfect=metrics(np.array([1,0,0]),rows)
        self.assertEqual(perfect['correct'],3);self.assertEqual(perfect['matched_pairs_correct'],1)
        wrong=metrics(np.array([1,1,0]),rows)
        self.assertEqual(wrong['negative_false_calls'],1);self.assertEqual(wrong['matched_pairs_correct'],0)
        self.assertEqual(wrong['by_template']['no']['correct'],0)
        self.assertEqual(wrong['by_language']['en']['correct'],1)

    def test_invalid_predictions_are_not_scored_as_abstention(self):
        row=[dict(id='a',pair=None,ready=0,template='a',language='en')]
        for bad in (np.array([2]),np.array([-1]),np.array([0,0]),np.array([.5])):
            with self.assertRaises(ValueError):metrics(bad,row)


if __name__=='__main__':unittest.main()
