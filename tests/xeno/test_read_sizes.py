"""#37: read_sizes.py turns the serve log's prompt-part lines into the histogram that decides what the dual-GPU prompt
path is worth in daily use (split + wave only runs parts of 2,048 tokens or more)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "perf"))
import read_sizes as rs  # noqa: E402

LOG = """strata generate: 13 expert-pool workers + the host thread
strata serve: prompt part root: 9800 tokens [0, 9800) of 14000 (batched) in 12000.0 ms
strata serve: prompt part history: 3000 tokens [9800, 12800) of 14000 (batched) in 3700.5 ms
strata serve: prompt part new turn: 1199 tokens [12800, 13999) of 14000 (batched) in 1500.0 ms
strata serve: prompt part new turn: 40 tokens [13999, 14039) of 14040 (windows) in 90.0 ms
unrelated line
"""


def test_every_part_line_is_parsed():
    parts = rs.parse(LOG)
    assert [p["kind"] for p in parts] == ["root", "history", "new turn", "new turn"]
    assert parts[0] == {"kind": "root", "tokens": 9800, "start": 0, "end": 9800, "total": 14000, "path": "batched",
                        "ms": 12000.0}


def test_the_histogram_counts_parts_and_prefill_time_per_bin():
    h = rs.histogram(rs.parse(LOG))
    assert h["< 2K"] == {"parts": 2, "tokens": 1239, "ms": 1590.0}
    assert h["2-4K"] == {"parts": 1, "tokens": 3000, "ms": 3700.5}
    assert h["4-8K"] == {"parts": 0, "tokens": 0, "ms": 0.0}
    assert h["> 8K"] == {"parts": 1, "tokens": 9800, "ms": 12000.0}


def test_split_eligible_share_is_the_prefill_time_of_parts_of_2048_tokens_or_more():
    # the split runs only chunks of 2,048+ tokens: 3000 and 9800 here, 15700.5 of 17290.5 ms
    share = rs.split_eligible_share(rs.parse(LOG))
    assert abs(share - 15700.5 / 17290.5) < 1e-9


REQ = """strata serve: request metrics: decode entries 39840 = primary 15482 + secondary 10174 + pcie 0 + cpu 14184; cpu experts 543.3 ms; nvme loads 0; private commit 38.41 GiB
strata serve: decode expert cache hit rate: 52.2% (15482 hits / 29666 lookups)
"""


def test_request_metrics_are_parsed_and_the_tiers_add_up():
    # #9 (PRD story 47): the per-request decode counters; the four tiers must add up to the routed entries
    (r,) = rs.parse_requests(REQ)
    assert r == {"entries": 39840, "primary": 15482, "secondary": 10174, "pcie": 0, "cpu": 14184, "cpu_ms": 543.3,
                 "nvme_loads": 0, "commit_gib": 38.41}
    assert r["primary"] + r["secondary"] + r["pcie"] + r["cpu"] == r["entries"]
