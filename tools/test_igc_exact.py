"""Independent integer oracles for typed native GPU arithmetic, not model tests."""
import os
from pathlib import Path
import random
import struct
import subprocess
import tempfile
import unittest

REQUEST = struct.Struct('<IIIIQQ64s64s')
RESULT = struct.Struct('<IIQQ64s')
SUCCESS, MALFORMED, UNSUPPORTED, OVERFLOW = range(4)


def operand(value):
    text = str(abs(value))
    if len(text) > 62:
        raise ValueError('test operand exceeds ABI')
    return bytes([len(text), int(value < 0), *map(int, reversed(text))]) + bytes(62 - len(text))


def request(a, b, op, ident=1, position=0, version=1, domain=1, reserved=0):
    return REQUEST.pack(version, domain, op, reserved, ident, position, operand(a), operand(b))


def expected(a, b, op):
    value = {1: lambda: a+b, 2: lambda: a-b, 3: lambda: a*b}[op]()
    return (OVERFLOW, None) if len(str(abs(value))) > 62 else (SUCCESS, value)


@unittest.skipUnless(os.environ.get('IGC_EXACT_RUN'), 'typed CUDA executor not selected')
class TypedExact(unittest.TestCase):
    def run_batch(self, records, outcomes, sequence=False):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp)/'in.bin', Path(tmp)/'out.bin'
            source.write_bytes(b''.join(records))
            command = [os.environ['IGC_EXACT_RUN'], str(source), str(target)]
            if sequence:
                command.append('--sequence')
            run = subprocess.run(command,
                                 capture_output=True, text=True, timeout=30, creationflags=0x08000000)
            self.assertEqual(run.returncode, 0, run.stdout+run.stderr)
            self.assertIn('repeat=bitwise', run.stdout)
            data = target.read_bytes()
            self.assertEqual(len(data), RESULT.size * len(records))
            for i, (record, outcome) in enumerate(zip(records, outcomes, strict=True)):
                version, status, ident, position, value = RESULT.unpack_from(data, i*RESULT.size)
                inputs = REQUEST.unpack(record)
                self.assertEqual((version, ident, position), (1, inputs[4], inputs[5]))
                self.assertEqual(status, outcome[0], (i, record.hex(), status, outcome))
                if status == SUCCESS:
                    length, negative = value[:2]
                    self.assertTrue(1 <= length <= 62)
                    digits = value[2:2+length]
                    self.assertTrue(all(d < 10 for d in digits))
                    actual = int(''.join(map(str, reversed(digits)))) * (-1 if negative else 1)
                    self.assertEqual(actual, outcome[1])
                    self.assertEqual(value, operand(actual))
                else:
                    self.assertEqual(value, bytes(64), 'failure must publish no numeric value')

    def test_signed_add_subtract_multiply(self):
        cases = [(a,b,op) for a,b in [(0,0),(17,23),(-17,23),(17,-23),(-17,-23),(99,1),(1,99)]
                 for op in [1,2,3]]
        self.run_batch([request(a,b,op,i+1) for i,(a,b,op) in enumerate(cases)],
                       [expected(a,b,op) for a,b,op in cases])

    def test_digit_capacity_carry_borrow_and_overflow(self):
        maximum = 10**62-1
        cases = [(maximum,0,1),(maximum,1,1),(-maximum,-1,1),(maximum,1,2),
                 (-maximum,1,2),(maximum,1,3),(maximum,-1,3),(maximum,0,3),
                 (10**31,10**31,3),(10**30,10**31,3),(10**61,1,2),
                 (10**61,1,1),(10**61,-10**61,1)]
        self.run_batch([request(a,b,op,i+1) for i,(a,b,op) in enumerate(cases)],
                       [expected(a,b,op) for a,b,op in cases])

    def test_frozen_heldout_signed_integer_corpus(self):
        rng = random.Random(20261029)
        cases = []
        for _ in range(256):
            a = rng.randrange(10**rng.randrange(1,63)) * rng.choice([-1,1])
            b = rng.randrange(10**rng.randrange(1,63)) * rng.choice([-1,1])
            cases.append((a,b,rng.choice([1,2,3])))
        for start in range(0,len(cases),64):
            batch = cases[start:start+64]
            self.run_batch([request(a,b,op,start+i+1,2**63+i) for i,(a,b,op) in enumerate(batch)],
                           [expected(a,b,op) for a,b,op in batch])

    def test_malformed_records_never_publish_zero_as_an_answer(self):
        base = request(17,23,1)
        invalid = []
        for offset, value in [(0,2),(12,1),(16,0),(32,0),(33,2),(34,10),(35,0),(36,1)]:
            item=bytearray(base);item[offset]=value;invalid.append(bytes(item))
        negative_zero=bytearray(request(0,1,1));negative_zero[33]=1;invalid.append(bytes(negative_zero))
        self.run_batch(invalid,[(MALFORMED,None)]*len(invalid))

    def test_unsupported_operation_and_domain_are_explicit(self):
        self.run_batch([request(1,2,4),request(1,2,1,domain=2)],[(UNSUPPORTED,None)]*2)

    def test_reused_output_drops_old_digits_and_failed_values(self):
        cases = [(10**61,1,1),(0,0,3),(10**61,10**61,3),(17,23,3)]
        self.run_batch([request(a,b,op,i+1) for i,(a,b,op) in enumerate(cases)],
                       [expected(a,b,op) for a,b,op in cases], sequence=True)

    def test_binary_size_and_batch_limits_are_rejected(self):
        for content in [b'',b'x',request(1,2,1)*65]:
            with self.subTest(bytes=len(content)), tempfile.TemporaryDirectory() as tmp:
                source,target=Path(tmp)/'in.bin',Path(tmp)/'out.bin'
                source.write_bytes(content)
                run=subprocess.run([os.environ['IGC_EXACT_RUN'],str(source),str(target)],
                                   capture_output=True,text=True,timeout=30,creationflags=0x08000000)
                self.assertEqual(run.returncode,2,run.stdout+run.stderr)
                self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main()
