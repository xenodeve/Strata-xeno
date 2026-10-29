"""Offline Fraction oracle for ABI3; never imported by the native runtime.

Decimal inputs are exact coefficients/scales, not floats. No rounded decimal
format is inferred. These numeric cases are unrelated to learned acceptance64.
"""
from __future__ import annotations
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import struct

SEED = 0x1BF20261029
LIMIT = 10**62
HEADER = struct.Struct("<IIIIQQ")
OPERAND = struct.Struct("<64s64sII")
RESULT = struct.Struct("<IIQQ64s64s")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def integer(value: int) -> bytes:
    digits = str(abs(value))
    if len(digits) > 62:
        raise ValueError("operand_range")
    return bytes((len(digits), int(value < 0))) + bytes(map(int, reversed(digits))) + bytes(62-len(digits))


def expected(domain: int, operation: int, an: int, ad: int, bn: int, bd: int,
             sa: int, sb: int) -> tuple[int, Fraction | None]:
    if domain not in (2, 3) or operation not in (1, 2, 3, 4) or (domain == 3 and operation == 4):
        return 2, None
    if domain == 3:
        if ad != 1 or bd != 1:
            return 1, None
        if sa > 61 or sb > 61:
            return 2, None
        a, b = Fraction(an, 10**sa), Fraction(bn, 10**sb)
    else:
        if sa or sb:
            return 1, None
        if not ad or not bd:
            return 5, None
        a, b = Fraction(an, ad), Fraction(bn, bd)
    if operation == 4 and not b:
        return 5, None
    value = a+b if operation == 1 else a-b if operation == 2 else a*b if operation == 3 else a/b
    if abs(value.numerator) >= LIMIT or value.denominator >= LIMIT:
        return 3, None
    return 0, value


def generate(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    records, cases = [], []

    def add(name, domain, op, an, ad, bn, bd, sa=0, sb=0, *, version=3,
            reserved=0, owner=None, mutation=None, status=None):
        invocation = len(cases)+1 if owner is None else owner
        position = 2**63+len(cases)
        record = bytearray(HEADER.pack(version, domain, op, reserved, invocation, position)
                           + OPERAND.pack(integer(an), integer(ad), sa, 0)
                           + OPERAND.pack(integer(bn), integer(bd), sb, 0))
        if mutation:
            record[mutation[0]] = mutation[1]
        code, value = expected(domain, op, an, ad, bn, bd, sa, sb) if status is None else (status, None)
        result = RESULT.pack(3, code, invocation, position,
                             integer(value.numerator) if value is not None else bytes(64),
                             integer(value.denominator) if value is not None else bytes(64))
        records.append(bytes(record))
        cases.append(dict(name=name, domain=domain, operation=op, lhs=[str(an), str(ad), sa],
                          rhs=[str(bn), str(bd), sb], status=code,
                          value=None if value is None else str(value), expected_hex=result.hex()))

    edge = [(0,1),(1,1),(-1,1),(1,3),(2,4),(-7,3),(7,-3),
            (LIMIT-1,LIMIT-1),(LIMIT-1,LIMIT-2),(10**61,1),(1,10**61)]
    for op in (1,2,3,4):
        for an,ad in edge:
            for bn,bd in edge:
                add("rational_cross",2,op,an,ad,bn,bd)
    decimals = [(0,0),(1,1),(2,1),(-1,1),(25,2),(LIMIT-1,0),(LIMIT-1,61),(1,61),(-1,61),(10**61,61)]
    for op in (1,2,3):
        for an,sa in decimals:
            for bn,sb in decimals:
                add("decimal_cross",3,op,an,1,bn,1,sa,sb)
    # Literal known values and cancellation beyond the output digit capacity.
    add("one_third_plus_sixth",2,1,1,3,1,6)
    add("point_one_plus_point_two",3,1,1,1,2,1,1,1)
    add("wide_sum_reduce",2,1,LIMIT-2,LIMIT-1,1,LIMIT-1)
    add("wide_product_reduce",2,3,LIMIT-1,5,5,LIMIT-1)
    add("wide_self_divide",2,4,LIMIT-1,LIMIT-2,LIMIT-1,LIMIT-2)
    rng = random.Random(SEED)

    def signed():
        limit = 10**rng.randint(1,62)
        return rng.randrange(-limit+1, limit)

    for op in (1,2,3,4):
        for i in range(64):
            add(f"rational_seeded_{i}",2,op,signed(),signed(),signed(),signed())
    for op in (1,2,3):
        for i in range(64):
            add(f"decimal_seeded_{i}",3,op,signed(),1,signed(),1,rng.randrange(62),rng.randrange(62))
    for op in (1,2,3,4):
        for version in (0,1,2,4,2**32-1):
            add("foreign_version",2,op,7,3,1,6,version=version,status=1)
        add("reserved_header",2,op,7,3,1,6,reserved=1,status=1)
        add("zero_owner",2,op,7,3,1,6,owner=0,status=1)
        for operand in (32,168):
            for component in (0,64):
                for offset,byte in ((0,0),(0,63),(1,2),(2,10),(3,1),(0,2)):
                    add("malformed_integer",2,op,7,3,1,6,mutation=(operand+component+offset,byte),status=1)
            add("reserved_operand",2,op,7,3,1,6,mutation=(operand+132,1),status=1)
            add("rational_scale",2,op,7,3,1,6,mutation=(operand+128,1),status=1)
        add("negative_zero_numerator",2,op,0,3,1,6,mutation=(33,1),status=1)
        add("negative_zero_denominator",2,op,7,0,1,6,mutation=(97,1),status=1)
    for op in (1,2,3):
        for denominator in (0,2,-1):
            add("decimal_nonunit_denominator",3,op,1,denominator,2,1,1,1,status=1)
        for scale in (62,2**32-1):
            add("unsupported_scale",3,op,1,1,2,1,scale,1,status=2)
    add("decimal_division",3,4,1,1,2,1,1,1,status=2)
    add("unknown_domain",99,1,7,3,1,6,status=2)
    add("unknown_operation",2,99,7,3,1,6,status=2)
    request_data = b"".join(records)
    result_data = b"".join(bytes.fromhex(c["expected_hex"]) for c in cases)
    (directory/"requests.bin").write_bytes(request_data)
    (directory/"expected.bin").write_bytes(result_data)
    (directory/"cases.json").write_text(json.dumps(cases,indent=2)+"\n",encoding="utf-8")
    manifest = dict(schema=1,abi=3,seed=SEED,count=len(cases),oracle="Python fractions.Fraction; exact, no rounding",
                    request_sha256=sha(request_data),expected_sha256=sha(result_data),
                    generator_sha256=sha(Path(__file__).read_bytes()),
                    status_counts=dict(Counter(str(c["status"]) for c in cases)))
    (directory/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(manifest))


def verify(directory: Path, actual_path: Path) -> None:
    manifest = json.loads((directory/"manifest.json").read_text())
    request_data, result_data = (directory/"requests.bin").read_bytes(), (directory/"expected.bin").read_bytes()
    if sha(request_data)!=manifest["request_sha256"] or sha(result_data)!=manifest["expected_sha256"]:
        raise ValueError("frozen_corpus_changed")
    actual = actual_path.read_bytes()
    if actual != result_data:
        cases = json.loads((directory/"cases.json").read_text())
        errors = [dict(index=i,**c,actual_hex=actual[i*RESULT.size:(i+1)*RESULT.size].hex())
                  for i,c in enumerate(cases) if actual[i*RESULT.size:(i+1)*RESULT.size]!=result_data[i*RESULT.size:(i+1)*RESULT.size]]
        print(json.dumps(dict(pass_=False,actual_bytes=len(actual),expected_bytes=len(result_data),errors=errors[:8])))
        raise SystemExit(1)
    print(json.dumps(dict(pass_=True,count=manifest["count"],actual_sha256=sha(actual),request_sha256=manifest["request_sha256"])))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("generate","verify"));parser.add_argument("directory",type=Path)
    parser.add_argument("actual",type=Path,nargs="?");args=parser.parse_args()
    if args.action=="generate":
        if args.actual:parser.error("generate takes no actual output")
        generate(args.directory)
    else:
        if not args.actual:parser.error("verify requires actual output")
        verify(args.directory,args.actual)


if __name__=="__main__":
    main()
