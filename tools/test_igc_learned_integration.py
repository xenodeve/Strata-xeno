import struct
import unittest
from tools.igc.learned_integration import parse_audit


def integer(value):
    digits = bytes(map(int, reversed(str(abs(value)))))
    return bytes([len(digits), int(value < 0)]) + digits + bytes(62 - len(digits))


def success():
    header = struct.pack("<QQ6I", 3, 7, 0, 1, 1, 1, 0, 0)
    request = struct.pack("<IIIIQQ64s64s", 1, 1, 3, 0, 3, 7, integer(31), integer(-24))
    result = struct.pack("<IIQQ64s", 1, 0, 3, 7, integer(-744))
    return header + request + result


class LearnedAudit(unittest.TestCase):
    def test_success_preserves_exact_execution_evidence(self):
        result = parse_audit(success())
        self.assertEqual(result["request"], [3, 31, -24])
        self.assertEqual(result["result"], -744)
        self.assertEqual(result["evaluations"], 1)

    def test_rejects_malformed_or_wrong_owner_evidence(self):
        for offset, value in ((36, 1), (208, 99), (224, 0)):
            with self.subTest(offset=offset):
                raw = bytearray(success())
                struct.pack_into("<I", raw, offset, value)
                with self.assertRaises(ValueError):
                    parse_audit(raw)
        with self.assertRaises(ValueError):
            parse_audit(success()[:-1])

    def test_failure_cannot_contain_a_numeric_answer(self):
        raw = bytearray(success())
        struct.pack_into("<I", raw, 24, 0)  # no execution
        struct.pack_into("<I", raw, 28, 3)  # rejected
        struct.pack_into("<I", raw, 204, 4)  # fault result
        with self.assertRaises(ValueError):
            parse_audit(raw)
        raw[224:288] = bytes(64)
        result = parse_audit(raw)
        self.assertIsNone(result["result"])


if __name__ == "__main__":
    unittest.main()
