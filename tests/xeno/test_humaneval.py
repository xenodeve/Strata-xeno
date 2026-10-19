"""The task-quality gate (tests/xeno/perf/humaneval.py): code extraction, the isolated base and plus checks, and the
gained / lost comparison.  No server, no GPU: two hand-written problems stand in for HumanEval's."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "perf"))
import humaneval  # noqa: E402

PROB = {
    "task_id": "T/0",
    "prompt": "def add(a, b):\n    \"\"\"a + b\"\"\"\n",
    "entry_point": "add",
    "canonical_solution": "    return a + b\n",
    "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(-1, 1) == 0\n",
    "base_input": [[1, 2], [-1, 1]],
    "plus_input": [[10 ** 6, 1], [0.1, 0.2]],
    "atol": 0,
}


def test_extract_takes_the_first_fenced_block():
    reply = "Here:\n```python\ndef add(a, b):\n    return a + b\n```\nand\n```python\nx = 1\n```"
    assert humaneval.extract(reply) == "def add(a, b):\n    return a + b\n"
    assert humaneval.extract("```\nprint(1)\n```") == "print(1)\n"
    assert humaneval.extract("no fence") == "no fence"


def test_base_check_passes_a_right_answer_and_fails_a_wrong_one(tmp_path):
    ok, _ = humaneval.check_base(PROB, "def add(a, b):\n    return a + b\n", str(tmp_path))
    assert ok
    bad, why = humaneval.check_base(PROB, "def add(a, b):\n    return a - b\n", str(tmp_path))
    assert not bad and "AssertionError" in why


def test_plus_check_compares_with_the_canonical_solution_on_every_input(tmp_path):
    ok, _ = humaneval.check_plus(PROB, "def add(a, b):\n    return b + a\n", str(tmp_path))
    assert ok   # 0.1 + 0.2 within the float tolerance either way round
    # right on the base tests, wrong on a plus input: HumanEval passes it, HumanEval+ must not
    wrong_plus = "def add(a, b):\n    return a + b if a < 100 else 0\n"
    assert humaneval.check_base(PROB, wrong_plus, str(tmp_path))[0]
    ok_plus, why = humaneval.check_plus(PROB, wrong_plus, str(tmp_path))
    assert not ok_plus and "MISMATCH" in why


def test_compare_counts_gained_and_lost_problems():
    lines = humaneval.compare({"base": {"a": True, "b": True, "c": False},
                               "arm": {"a": True, "b": False, "c": True}})
    assert lines[0].startswith("base: pass@1 2/3")
    assert lines[1].endswith("vs base: +1 -1 (net +0)")


def test_long_context_prefix(tmp_path):
    f = tmp_path / "ctx.txt"
    f.write_text("abcdef" * 10, encoding="utf-8")
    assert humaneval.long_context(None, 0) == []
    m = humaneval.long_context(str(f), 9)
    assert len(m) == 1 and m[0]["role"] == "system" and m[0]["content"].endswith("abcdefabc")
