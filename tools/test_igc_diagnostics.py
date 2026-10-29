"""Independent operand order, padding and little-endian diagnostic controls."""
import unittest
from tools.igc.readout import targets
from tools.igc.diagnostics import canonical_value,analyze


class DiagnosticAccounting(unittest.TestCase):
    def test_sign_order_and_reversal_are_distinguished(self):
        rows=[dict(ready_op=2,lhs=-17,rhs=23),dict(ready_op=2,lhs=7,rhs=0)]
        y=targets(rows)
        self.assertEqual(canonical_value(y[0],0),-17)
        self.assertEqual(canonical_value(y[0],1),23)
        y[0,5],y[0,6]=y[0,6],y[0,5]
        report=analyze(rows,y)
        self.assertEqual(report['lhs_exact'],1)
        self.assertEqual(report['rhs_exact'],2)
        self.assertEqual(report['lhs_digit_reversal'],1)
        self.assertEqual(report['operand_swaps'],0)
        self.assertEqual(report['lhs_tens_nonpadding'],dict(correct=0,n=1))

    def test_swapped_subtraction_and_noncanonical_zero_are_not_success(self):
        rows=[dict(ready_op=2,lhs=17,rhs=-23)]
        y=targets([dict(ready_op=2,lhs=-23,rhs=17)])
        self.assertEqual(analyze(rows,y)['operand_swaps'],1)
        zero=targets([dict(ready_op=1,lhs=0,rhs=7)])[0]
        for field,value in [(1,1),(3,1),(6,2)]:
            changed=zero.copy();changed[field]=value
            self.assertIsNone(canonical_value(changed,0))


if __name__=='__main__':unittest.main()
