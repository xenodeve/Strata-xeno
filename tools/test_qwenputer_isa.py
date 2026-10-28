"""Independent expected integer outcomes and fault records for circuit tests."""
import unittest

from tools.qwenputer.isa import instruction_cases, register_value
from tools.qwenputer.tokenizer import NativeInputError


class IsaTests(unittest.TestCase):
    def test_multiple_channel_columns_cannot_hide_an_invalid_register_bit(self):
        columns = {b: [2 + b] for b in range(32)}
        columns[0] = [2, 40]
        row = [0.0] * 128
        row[2] = 128
        self.assertEqual(register_value(row, columns), 1)
        row[40] = 128
        with self.assertRaisesRegex(NativeInputError, 'register_bit'):
            register_value(row, columns)

    def test_all_major_instruction_groups_and_all_eight_faults_are_defined(self):
        cases = instruction_cases()
        self.assertGreaterEqual(len(cases), 40)
        self.assertEqual({c['fault'] for c in cases if c.get('fault') is not None}, set(range(8)))
        by_name = {c['name']: c for c in cases}
        self.assertEqual(by_name['add-overflow']['expected_register3'], 0x80000000)
        self.assertEqual(by_name['mulh-signed']['expected_register3'], 0xffffffff)
        self.assertEqual(by_name['mulhu']['expected_register3'], 2)
        self.assertEqual(by_name['load-lb']['expected_register3'], 0xffffff80)
        self.assertEqual(by_name['load-lbu']['expected_register3'], 128)
        self.assertEqual(by_name['store-byte']['expected_register3'], 0xaabb11dd)
        self.assertEqual(by_name['jal']['expected_register3'], 0x40004)
        self.assertEqual(by_name['refresh']['initial_n'], 63)
        self.assertEqual(by_name['putchar']['record'].count(32768 + 93), 2)


if __name__ == '__main__':
    unittest.main()
