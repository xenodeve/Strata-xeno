"""Readout accounting checks that guard train/heldout and exact operand fidelity."""
import unittest
import numpy as np
from tools.igc.readout import targets, metrics, standardize


class ReadoutAccounting(unittest.TestCase):
    def rows(self):
        return [dict(ready_op=3,lhs=-17,rhs=23),dict(ready_op=0,lhs=88,rhs=99)]

    def test_perfect_prediction_and_one_wrong_digit_are_distinguished(self):
        y=targets(self.rows())
        report=metrics(y,self.rows())
        self.assertEqual(report['positive_exact_routed'],1)
        self.assertEqual(report['negative_false_calls'],0)
        y[0,5]=8  # ones digit of lhs, operator stays correct
        report=metrics(y,self.rows())
        self.assertEqual(report['ready_operator_correct'],2)
        self.assertEqual(report['positive_exact_routed'],0)

    def test_negative_calls_and_noncanonical_zero_do_not_count_as_exact(self):
        y=targets(self.rows())
        y[1,0]=1
        self.assertEqual(metrics(y,self.rows())['negative_false_calls'],1)
        zero=[dict(ready_op=1,lhs=0,rhs=23)]
        y=targets(zero)
        y[0,1]=1
        self.assertEqual(metrics(y,zero)['positive_exact_routed'],0)

    def test_normalization_uses_only_training_rows(self):
        x=np.array([[1.,2.],[3.,4.],[100000.,100000.]],dtype=np.float32)
        normalized,mean,scale=standardize(x,np.array([True,True,False]))
        np.testing.assert_array_equal(mean,[2,3])
        np.testing.assert_array_equal(scale,[1,1])
        np.testing.assert_array_equal(normalized[0],[-1,-1])
        with self.assertRaises(ValueError):
            standardize(x,np.array([False,False,False]))


if __name__=='__main__':
    unittest.main()
