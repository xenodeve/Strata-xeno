import unittest
from tools.igc.alignment import digit_positions


class BytesTokenizer:
    def decode_bytes(self,ids):return bytes(ids)


class Alignment(unittest.TestCase):
    def test_labels_bind_actual_numeric_token_positions_and_skip_unused_tens(self):
        row=dict(expression='(-3) * (17)',lhs=-3,rhs=17)
        raw=b'S (-3) * (17) A\n';tokens=list(raw)
        labels=digit_positions(row,tokens,BytesTokenizer())
        self.assertEqual(labels[1],-1)
        anchor=len(tokens)-1
        self.assertEqual([chr(tokens[p+anchor-80]) for p in labels if p>=0],['3','7','1'])
    def test_ambiguous_or_wrong_label_is_not_silently_annotated(self):
        row=dict(expression='3+17',lhs=3,rhs=17)
        for raw in (b'3+17 and3+17\n',b'4+17\n'):
            with self.assertRaises(ValueError):digit_positions(row,list(raw),BytesTokenizer())
        with self.assertRaises(ValueError):digit_positions({**row,'lhs':4},list(b'3+17\n'),BytesTokenizer())
    def test_binary_subtraction_is_not_an_operand_sign(self):
        for expression,rhs in [('93-3',3),('93--3',-3),('(93) - (-3)',-3),('93 - 3',3)]:
            raw=('S '+expression+' A\n').encode('ascii')
            positions=digit_positions(dict(expression=expression,lhs=93,rhs=rhs),list(raw),BytesTokenizer())
            self.assertEqual([chr(raw[p+len(raw)-1-80]) for p in positions if p>=0],['3','9','3'])


if __name__=='__main__':unittest.main()
