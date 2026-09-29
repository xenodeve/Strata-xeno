"""Guards the latency parser against Strata's real counter lines (copied from run
strata-claude-abba-thai-2-n4, 2026-09-28).  A parser that silently misses a line would
report a plausible budget with a stage missing, so every field is asserted."""
import unittest

from tests.xeno.latency_breakdown import budget, parse

STDOUT = """speculation              173 rounds of 4, drafts accepted 83 of 171 (0.485), 1.48 tokens per round
verify window            wait for rings 21.518  pool 21.226  host 0.910  commit 0.773 ms/round; CPU experts 4.65 distinct / 5.57 routed per layer
pool multi               gate/up 7.997  quantize 0.120  down 4.090 ms/round; 25.5 GB/s over the rows phases; CPU pool call 21.215 ms/round
dispatch                 plan 2.832  activation quantize 0.168  jobs 0.359  run 17.844 ms/round
mtp                      1.406 ms/round drafting (172 rounds), MTP prompt 10.3 ms, 869 MiB of VRAM
decode                   256 tokens in 8303.7 ms  ->  30.83 tok/s
"""
STDERR = ("strata generate: prefill 3964 tokens in 2 chunks, 13677.6 ms (289.8 tok/s); experts streamed 24475 "
          "(0 by DMA, host 7190.6 ms), resident 9676; PLE 794.6 ms\n")


class LatencyBreakdownTest(unittest.TestCase):
    def test_parses_opt_in_secondary_stage_timings(self):
        timed = STDOUT + (
            "secondary host           plan 0.420 switch 0.250 enqueue 3.700 query 0.080 copyout 0.160 ms/round\n"
            "secondary device         H2D 0.730 clear 0.100 quantize 0.190 expert 2.100 D2H 0.330 ms/round; 2784 launches\n"
            "secondary D2H bytes      full 100.00 MiB requested 25.00 MiB\n"
        )
        p = parse(timed)
        self.assertAlmostEqual(p['secondary_host_plan'], 0.420)
        self.assertAlmostEqual(p['secondary_host_switch'], 0.250)
        self.assertAlmostEqual(p['secondary_host_enqueue'], 3.700)
        self.assertAlmostEqual(p['secondary_host_query'], 0.080)
        self.assertAlmostEqual(p['secondary_host_copyout'], 0.160)
        self.assertAlmostEqual(p['secondary_h2d'], 0.730)
        self.assertAlmostEqual(p['secondary_clear'], 0.100)
        self.assertAlmostEqual(p['secondary_quantize'], 0.190)
        self.assertAlmostEqual(p['secondary_expert'], 2.100)
        self.assertAlmostEqual(p['secondary_d2h'], 0.330)
        self.assertEqual(p['secondary_launches'], 2784)
        self.assertAlmostEqual(p['secondary_d2h_full_mib'], 100.0)
        self.assertAlmostEqual(p['secondary_d2h_requested_mib'], 25.0)

    def test_dispatch_run_separates_cpu_pool_and_secondary_finish(self):
        timed = STDOUT + "dispatch detail          CPU pool 18.250 secondary finish 5.125 ms/round\n"
        p = parse(timed)
        self.assertAlmostEqual(p['cpu_pool_self'], 18.250)
        self.assertAlmostEqual(p['secondary_finish_total'], 5.125)

    def test_parses_prefill_sources_per_tier(self):
        # #5 P5a: where the prompt's streamed experts came from, so 4070-assisted prefill is sized from a counter
        err = STDERR + ("strata generate: prefill sources pinned 100 (0.13 GB) pageable 20000 (27.65 GB) "
                        "peer 4375 (6.05 GB) nvme 0 (0.00 GB)\n")
        p = parse(STDOUT, err)
        self.assertEqual(p['prefill_src_pinned'], 100)
        self.assertEqual(p['prefill_src_pageable'], 20000)
        self.assertEqual(p['prefill_src_peer'], 4375)
        self.assertEqual(p['prefill_src_nvme'], 0)
        self.assertAlmostEqual(p['prefill_src_peer_gb'], 6.05)
        self.assertAlmostEqual(p['prefill_src_pageable_gb'], 27.65)

    def test_parses_every_counter(self):
        p = parse(STDOUT, STDERR)
        self.assertEqual(p['rounds'], 173)
        self.assertAlmostEqual(p['decode_ms'], 8303.7)
        self.assertAlmostEqual(p['wait_rings'], 21.518)
        self.assertAlmostEqual(p['pool'], 21.226)
        self.assertAlmostEqual(p['cpu_distinct'], 4.65)
        self.assertAlmostEqual(p['pool_gateup'], 7.997)
        self.assertAlmostEqual(p['disp_plan'], 2.832)
        self.assertAlmostEqual(p['mtp'], 1.406)
        self.assertAlmostEqual(p['prefill_host_ms'], 7190.6)
        self.assertAlmostEqual(p['prefill_ple_ms'], 794.6)

    def test_budget_rows_sum_to_round(self):
        rows = dict((name, ms) for name, ms, _ in budget(parse(STDOUT)))
        total = rows.pop('round total')
        parts = sum(ms for name, ms in rows.items() if not name.startswith(' '))
        self.assertAlmostEqual(parts, total, places=6)

    def test_budget_names_actual_host_intervals(self):
        rows = dict((name, ms) for name, ms, _ in budget(parse(STDOUT)))
        self.assertAlmostEqual(rows['Host waits for primary GPU doorbell'], 21.518)
        self.assertAlmostEqual(rows['Host services expert callback'], 21.226)
        self.assertIn('Other/unaccounted wall time', rows)
        self.assertNotIn('GPU waits for CPU experts (wait for rings)', rows)
        self.assertNotIn('GPU compute + unaccounted', rows)

    def test_missing_decode_gives_no_budget(self):
        self.assertEqual(budget(parse('speculation 10 rounds of 3\n')), [])


if __name__ == '__main__':
    unittest.main()
