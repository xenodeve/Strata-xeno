import struct
import unittest
from tools.igc.output_fixture import pack_fixture,pack_request_fixture


class OutputFixture(unittest.TestCase):
    def test_v2_contains_typed_request_and_no_expected_result_record(self):
        record=dict(sample='a',ids=[1,2,3],invocation=7,invocation_position=2,position=2,output_offset=0)
        raw=pack_request_fixture([record],{'a':dict(operation=3,lhs=93,rhs=-3)},bytes(32),bytes(32),bytes(32))
        self.assertEqual(raw[:8],b'IGCOFX02');self.assertEqual(len(raw),300)
        fields=struct.unpack('<IIIIQQ64s64s',raw[140:])
        self.assertEqual(fields[:6],(1,1,3,0,7,2));self.assertEqual(fields[6][:4],bytes([2,0,3,9]));self.assertEqual(fields[7][:3],bytes([1,1,3]))
        with self.assertRaises(ValueError):pack_request_fixture([record],{'a':dict(operation=4,lhs=1,rhs=2)},bytes(32),bytes(32),bytes(32))
    def test_only_complete_initial_prefix_and_checked_result_are_serialized(self):
        record=dict(ids=[1,2,3],invocation=7,invocation_position=2,position=2,output_offset=0,
            checked_result_hex=struct.pack('<IIQQ64s',1,0,7,2,bytes([1,0,4])+bytes(61)).hex())
        raw=pack_fixture([record],bytes(32),bytes([1])*32,bytes([2])*32)
        self.assertEqual(raw[:12],b'IGCOFX01'+struct.pack('<I',1));self.assertEqual(len(raw),228)
        for replacement in [dict(position=3),dict(invocation_position=1),dict(output_offset=1),dict(ids=[1,-1,3])]:
            with self.assertRaises(ValueError):pack_fixture([{**record,**replacement}],bytes(32),bytes(32),bytes(32))
        with self.assertRaises(ValueError):pack_fixture([record,record],bytes(32),bytes(32),bytes(32))


if __name__=='__main__':unittest.main()
