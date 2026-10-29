"""Offline stdlib oracle for the separate ABI2 integer component.

Native execution consumes only requests.bin. No expected answer, Python
arithmetic or parser participates in its CUDA execution path. This is new
numeric component coverage, unrelated to the sealed learned-model acceptance64.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import struct

SEED = 0x1B20261029
LIMIT = 10**62
REQUEST = struct.Struct("<IIIIQQ64s64s")
RESULT = struct.Struct("<IIQQ64s")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def integer(value: int) -> bytes:
    digits = str(abs(value))
    if len(digits) > 62:
        raise ValueError("operand_range")
    return bytes((len(digits), int(value < 0))) + bytes(map(int, reversed(digits))) + bytes(62 - len(digits))


def answer(op: int, a: int, b: int) -> tuple[int, int | None]:
    """Python big integers/divmod/math.gcd, not a port of the digit kernel."""
    if op in (1, 2):
        if b == 0:
            return 5, None
        q, r = divmod(a, b)
        value = q if op == 1 else r
        assert a == b * q + r and abs(r) < abs(b)
    elif op == 3:
        value = math.gcd(a, b)
    elif op == 4:
        if a == 0 and b < 0:
            return 5, None
        if b < 0 and abs(a) > 1:
            return 2, None  # rational extension not part of this integer ABI
        if abs(a) <= 1:
            value = 1 if b == 0 else (0 if a == 0 else (-1 if a == -1 and b % 2 else 1))
        else:
            # Avoid constructing astronomical oracle integers. For |a|>=2,
            # a positive exponent above 1000 certainly exceeds the output range.
            value = pow(a, min(b, 1000))
    else:
        return 2, None
    return (3, None) if abs(value) >= LIMIT else (0, value)


def generate(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    cases: list[dict] = []
    requests: list[bytes] = []

    def add(name: str, op: int, a: int, b: int, *, mutation: tuple[int, int] | None = None,
            status: int | None = None, version: int = 2, domain: int = 1,
            reserved: int = 0, owner: int | None = None) -> None:
        invocation = len(cases) + 1 if owner is None else owner
        position = 2**63 + len(cases)
        raw = bytearray(REQUEST.pack(version, domain, op, reserved, invocation, position, integer(a), integer(b)))
        if mutation:
            raw[mutation[0]] = mutation[1]
        expected_status, value = answer(op, a, b) if status is None else (status, None)
        expected = RESULT.pack(2, expected_status, invocation, position, integer(value) if value is not None else bytes(64))
        cases.append(dict(name=name, operation=op, lhs=str(a), rhs=str(b), status=expected_status,
                          value=None if value is None else str(value), expected_hex=expected.hex()))
        requests.append(bytes(raw))

    edge = [-(LIMIT - 1), -(10**61), -99, -7, -1, 0, 1, 2, 3, 7, 99, 10**61, LIMIT - 1]
    for op in (1, 2, 3):
        for a in edge:
            for b in edge:
                add("signed_range_cross", op, a, b)
    for a in [-(LIMIT - 1), -10, -3, -2, -1, 0, 1, 2, 3, 10, LIMIT - 1]:
        for b in [-(LIMIT - 1), -207, -3, -1, 0, 1, 2, 3, 5, 61, 62, 205, 206, 207, LIMIT - 1]:
            add("power_boundary", 4, a, b)
    # Consecutive Fibonacci numbers stress Euclidean iteration count, unlike
    # only random operands. Both stay within the full 62-digit input range.
    a, b = 0, 1
    while a + b < LIMIT:
        a, b = b, a + b
    add("fibonacci_euclid", 3, a, b)
    add("fibonacci_euclid_swapped", 3, b, a)
    rng = random.Random(SEED)
    for op in (1, 2, 3, 4):
        for i in range(128):
            digits = rng.randint(1, 62)
            a = rng.randrange(-(10**digits - 1), 10**digits)
            # Vary both lengths independently: a fixed full-range rhs would
            # strongly favor |lhs|<|rhs| and trivial zero quotients.
            if op == 4:
                b = rng.randint(-10, 250)
            else:
                rhs_digits = rng.randint(1, 62)
                b = rng.randrange(-(10**rhs_digits - 1), 10**rhs_digits)
            add(f"seeded_{i}", op, a, b)
    for op in (1, 2, 3, 4):
        for version in (0, 1, 3, 2**32 - 1):
            add("foreign_version", op, 7, 3, version=version, status=1)
        add("reserved", op, 7, 3, reserved=1, status=1)
        add("zero_owner", op, 7, 3, owner=0, status=1)
        add("foreign_domain", op, 7, 3, domain=2, status=2)
        for operand_offset in (32, 96):
            for offset, byte in [(0, 0), (0, 63), (1, 2), (2, 10), (3, 1)]:
                add("malformed_integer", op, 7, 3, mutation=(operand_offset + offset, byte), status=1)
            add("leading_zero", op, 7, 3, mutation=(operand_offset, 2), status=1)
        add("negative_zero_lhs", op, 0, 3, mutation=(33, 1), status=1)
        add("negative_zero_rhs", op, 7, 0, mutation=(97, 1), status=1)
    add("unknown_operation", 999, 7, 3, status=2)
    request_bytes = b"".join(requests)
    expected_bytes = b"".join(bytes.fromhex(c["expected_hex"]) for c in cases)
    (directory / "requests.bin").write_bytes(request_bytes)
    (directory / "expected.bin").write_bytes(expected_bytes)
    (directory / "cases.json").write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")
    manifest = dict(schema=1, abi=2, seed=SEED, count=len(cases), request_bytes=len(request_bytes),
                    oracle="Python int/divmod/math.gcd/pow", request_sha256=digest(request_bytes),
                    expected_sha256=digest(expected_bytes), generator_sha256=digest(Path(__file__).read_bytes()),
                    status_counts=dict(Counter(str(c["status"]) for c in cases)))
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest))


def verify(directory: Path, actual_path: Path) -> None:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    requests = (directory / "requests.bin").read_bytes()
    expected = (directory / "expected.bin").read_bytes()
    if digest(requests) != manifest["request_sha256"] or digest(expected) != manifest["expected_sha256"]:
        raise ValueError("frozen_corpus_changed")
    actual = actual_path.read_bytes()
    if actual != expected:
        cases = json.loads((directory / "cases.json").read_text(encoding="utf-8"))
        errors = []
        for i, case in enumerate(cases):
            row = actual[i * RESULT.size:(i + 1) * RESULT.size]
            if row != expected[i * RESULT.size:(i + 1) * RESULT.size]:
                errors.append(dict(index=i, **case, actual_hex=row.hex()))
        print(json.dumps(dict(pass_=False, expected_bytes=len(expected), actual_bytes=len(actual), errors=errors[:12])))
        raise SystemExit(1)
    print(json.dumps(dict(pass_=True, count=manifest["count"], actual_sha256=digest(actual),
                          request_sha256=manifest["request_sha256"])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("generate", "verify"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("actual", type=Path, nargs="?")
    args = parser.parse_args()
    if args.action == "generate":
        if args.actual is not None:
            parser.error("generate takes no actual result")
        generate(args.directory)
    else:
        if args.actual is None:
            parser.error("verify requires the native output")
        verify(args.directory, args.actual)


if __name__ == "__main__":
    main()
