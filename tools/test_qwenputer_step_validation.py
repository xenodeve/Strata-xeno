"""Boundary checks for offline native test inputs, without CUDA or firmware."""
import struct
import unittest
from tools.qwenputer.step_validation import inputs_data


class NativeReplayInputs(unittest.TestCase):
    def test_tag_and_payload_are_distinct(self):
        self.assertEqual(inputs_data([230323, (1 << 24) + 248319], 248320),
                         struct.pack('<4I', 0, 230323, 1, 248319))

    def test_invalid_tokens_cannot_be_reinterpreted(self):
        for data in ([], [-1], [230324], [(1 << 24) + 248320], [True], [0.0], [1 << 32]):
            with self.subTest(data=data), self.assertRaises(ValueError):
                inputs_data(data, 248320)


if __name__ == '__main__':
    unittest.main()
