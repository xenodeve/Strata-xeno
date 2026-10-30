"""Offline qualification oracle, using synthetic fields and literal results only."""
import struct
import unittest
import numpy as np
from tools.igc.native_qualification import decisions, selection, expected_records, check_record


class NativeQualification(unittest.TestCase):
    def test_field_slices_and_ties(self):
        x=np.zeros((2,52),dtype=np.float32);x[1,3]=2;x[1,4+1]=3
        self.assertEqual(decisions(x).tolist(),[[0]*9,[3,1,0,0,0,0,0,0,0]])
        for bad in (x[:,:51],np.full((1,52),np.nan,dtype=np.float32)):
            with self.assertRaises(ValueError):decisions(bad)

    def test_original_gating_and_canonical_digits(self):
        a=[3,0,1,1,1,1,3,4,2]
        self.assertEqual(selection(a,a,1),(0,[3,31,-24]))
        self.assertEqual(selection(a,a,0),(1,None))
        b=a.copy();b[5]=2;self.assertEqual(selection(a,b,1),(4,None))
        z=[0]*9;self.assertEqual(selection(z,z,1),(1,None))
        for bad in ([1,0,0,0,0,1,3,2,0], [1,1,0,0,0,0,0,2,0], [1,0,0,1,0,1,0,2,0]):
            self.assertEqual(selection(bad,bad,1),(5,None))
        for bad in ([4]*9,[0]*8):
            with self.assertRaises(ValueError):selection(bad,bad,1)

    def test_literal_checked_request_and_result(self):
        request,result=expected_records(9,31,0,[3,31,-24])
        self.assertEqual(len(request),160);self.assertEqual(len(result),88)
        self.assertEqual(struct.unpack_from('<IIIIQQ',request),(1,1,3,0,9,31))
        self.assertEqual(result[:24],struct.pack('<IIQQ',1,0,9,31))
        self.assertEqual(result[24:29],bytes([3,1,4,4,7]))
        self.assertEqual(result[29:],bytes(59))

    def test_failure_has_no_stale_value(self):
        request,result=expected_records(12,4,4,None)
        self.assertEqual(request,struct.pack('<IIIIQQ',1,0,0,0,12,4)+bytes(128))
        self.assertEqual(result,struct.pack('<IIQQ',1,4,12,4)+bytes(64))
        for args in [(0,4,0,[1,2,3]),(1,262144,0,[1,2,3]),(1,4,1,[1,2,3]),(1,4,0,[1,100,2])]:
            with self.assertRaises(ValueError):expected_records(*args)

    def test_record_ownership_attempt_count_and_output(self):
        request,result=expected_records(3,20,0,[2,9,5]);out=np.arange(2560,dtype='<f4')
        raw=struct.pack('<QQ6I',3,20,0,1,1,1,0,0)+request+result+struct.pack('<I',0)+out.tobytes()
        np.testing.assert_array_equal(check_record(raw,3,20,0,[2,9,5]),out)
        for offset in (0,24,200,288):
            bad=bytearray(raw);bad[offset]^=1
            with self.assertRaises(ValueError):check_record(bytes(bad),3,20,0,[2,9,5])
        with self.assertRaises(ValueError):check_record(raw[:-1],3,20,0,[2,9,5])


if __name__=='__main__':unittest.main()
