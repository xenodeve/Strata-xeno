"""ab.py's per-stage parser: every speed claim in the reports is read through it (AGENTS.md: explain speed from
the same runs' per-stage counters), so a regex that silently misses a line would publish a wrong table."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "perf"))
import ab  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "ab_code_run.stdout")


def test_parse_run_reads_every_stage_from_a_real_run():
    output, stages = ab.parse_run(open(FIXTURE, encoding="utf-8").read())
    assert output.startswith("71093 12305 198 ")
    assert stages == {"tok/s": 81.17, "rings": 18.991, "pool": 14.462, "cpu_pool": 10.171, "sec_fin": 1.392,
                      "sec_launch": 1.846, "cpu_GBs": 26.6, "mtp": 2.398, "host": 1.037, "commit": 0.936,
                      "ttft_ms": 1655.3}


def test_parse_run_without_output_is_not_a_run():
    assert ab.parse_run("strata generate: the arena could not be reserved\n") == (None, {})
