"""Tests for tests/xeno/perf/ckpt_resume_check.py's log reader (#260)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "perf"))

import ckpt_resume_check as ckr  # noqa: E402

LOG = """\
strata serve: prompt part new turn: 16384 tokens [0, 16384) of 33247 (batched) in 8330.0 ms
strata serve: prompt part new turn: 16384 tokens [16384, 32768) of 33247 (batched) in 8456.5 ms
strata serve: prompt part new turn: 6 tokens [33240, 33246) of 33247 (windows) in 94.1 ms
strata serve: STATE_HASH_GDN 1576 0d2f 31ec
strata serve: STATE_HASH L=33247 gdn=9ea19051ac31371a ple=41a15074ecad16b6 tail=72 pooled=43 kv=740a0ef28420d721 mtp=e5
strata serve: prompt 33247 tokens = 0 reused + 33247 read in 18954 ms (1754.1 tok/s), 1 generated in 27 ms, 3 checkpoints
strata serve: prompt part shared: 6526 tokens [16384, 22910) of 23892 (batched) in 3439.3 ms
strata serve: STATE_HASH L=23892 gdn=8beb1d26dbff7ed0 ple=85 tail=e3 pooled=e1 kv=4012e988c34643d3 mtp=00
strata serve: prompt 23892 tokens = 16384 reused + 7508 read in 5436 ms (1381.1 tok/s), 1 generated in 17 ms
"""


def test_summarize_pairs_each_request_with_its_hash_and_parts():
    r = ckr.summarize(LOG.splitlines())
    assert len(r) == 2
    assert r[0]["tokens"] == 33247 and r[0]["reused"] == 0 and r[0]["ms"] == 18954
    assert r[0]["state"] == {"L": 33247, "gdn": "9ea19051ac31371a", "kv": "740a0ef28420d721"}
    assert r[0]["parts"] == [[0, 16384], [16384, 32768], [33240, 33246]]
    assert r[1]["reused"] == 16384 and r[1]["read"] == 7508
    assert r[1]["state"]["gdn"] == "8beb1d26dbff7ed0"
    assert r[1]["parts"] == [[16384, 22910]]


def test_summarize_without_hash_lines_keeps_the_prompt_line():
    r = ckr.summarize(["strata serve: prompt 100 tokens = 0 reused + 100 read in 50 ms (2000 tok/s)"])
    assert r == [{"tokens": 100, "reused": 0, "read": 100, "ms": 50, "state": None, "parts": []}]


def test_texts_share_the_start_and_differ_at_the_end(tmp_path):
    src = tmp_path / "src.cpp"
    src.write_text("".join(f"line {i}\n" for i in range(40000)), encoding="utf-8")
    r1, r2 = ckr.texts(str(src))
    assert r1[:78000] == r2[:78000] and len(r1) == 110000
    assert r2[78000:] != r1[78000:len(r2)]
