"""#18: the cross-boot determinism kit (tests/xeno/perf/cross_boot.py).  compare() must name the first token where two
runs diverge and every boot-dependent log fact that differs, since those are the candidate causes."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "perf"))
import cross_boot as cb  # noqa: E402


def run(tokens, facts):
    return {"tokens": tokens, "facts": facts}


def test_identical_runs_report_no_divergence():
    a = run([1, 2, 3], {"prefill sources": "pinned 0", "gpu free": "9.6 GiB"})
    d = cb.compare(a, a)
    assert d["first_divergent_token"] == -1
    assert d["facts"] == {}


def test_the_first_divergent_token_and_the_differing_facts_are_named():
    a = run([5, 6, 7, 8], {"prefill sources": "pageable 1928", "gpu free": "1.05 GiB", "cache": "6964 slots"})
    b = run([5, 6, 9, 8], {"prefill sources": "pageable 1923", "gpu free": "1.29 GiB", "cache": "6964 slots"})
    d = cb.compare(a, b)
    assert d["first_divergent_token"] == 2
    assert d["facts"] == {"prefill sources": ("pageable 1928", "pageable 1923"), "gpu free": ("1.05 GiB", "1.29 GiB")}


def test_a_shorter_output_diverges_where_it_ends():
    d = cb.compare(run([1, 2, 3], {}), run([1, 2], {}))
    assert d["first_divergent_token"] == 2


def test_facts_are_pulled_from_the_engine_log():
    stderr = ("strata generate: expert cache auto: 9.65 GiB free, 700 MiB reserved -> 6964 slots\n"
              "strata generate: prefill sources pinned 0 (0.00 GB) pageable 1923 (2.66 GB) peer 0 (0.00 GB)\n"
              "unrelated line\n")
    f = cb.facts_from(stderr, "")
    assert f["expert cache auto"] == "9.65 GiB free, 700 MiB reserved -> 6964 slots"
    assert f["prefill sources"].startswith("pinned 0 (0.00 GB) pageable 1923")
