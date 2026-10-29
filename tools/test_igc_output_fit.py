import unittest
from tools.igc.output_fit import score_predictions


class OutputFitMetrics(unittest.TestCase):
    def test_teacher_forced_sequence_and_numeric_accuracy_are_separate(self):
        records=[dict(sample='a',split='train',target=1,numeric_target=False),
                 dict(sample='a',split='train',target=2,numeric_target=True),
                 dict(sample='b',split='development',target=3,numeric_target=True)]
        result=score_predictions(records,[0,2,0],[1.,2.,3.])
        self.assertEqual(result['train']['numeric_correct'],1)
        self.assertEqual(result['train']['numeric_sequences_correct'],1)
        self.assertEqual(result['train']['complete_sequences_correct'],0)
        self.assertEqual(result['development']['numeric_sequences_correct'],0)
        self.assertEqual(result['train']['mean_cross_entropy'],1.5)
        with self.assertRaises(ValueError):score_predictions(records,[1],[])


if __name__=='__main__':unittest.main()
