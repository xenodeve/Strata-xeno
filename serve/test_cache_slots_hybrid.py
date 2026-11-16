"""serve/test_cache_slots_hybrid.py - the cache-slot rule "hybrid" (#321 round 4), no engine.

    python -m pytest serve/test_cache_slots_hybrid.py -q

A prompt placed in a slot reuses the prefix H it shares with the slot's last prompt now, and destroys that prompt's
part past H (len(last) - H), which its owner re-reads if it comes back.  hybrid joins a slot when the certain saving
is at least the possible loss (H >= JOIN * len(last), JOIN = 1/2), else takes a free slot, else evicts the least
recently used slot whose loss is at most the request's own size (else the least recently used slot).
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, HybridCacheSlots, MockEngine, Service  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def toks(tag, n):
    return [tag * 1_000_000 + i for i in range(n)]


MAIN_HEAD, CLS_ROOT, SUB_HEAD = toks(1, 10000), toks(2, 28000), toks(3, 12000)


class Hybrid(unittest.TestCase):
    def test_a_continuation_stays_in_its_slot(self):
        slots = HybridCacheSlots(4)
        conv = MAIN_HEAD + toks(10, 3000)
        self.assertEqual(slots.choose(conv), (0, None, "free"))
        for k in range(5):
            prev, conv = conv, conv + toks(20 + k, 2000)
            self.assertEqual(slots.choose(conv), (0, len(prev), "continues"))

    def test_another_agents_classifier_joins_the_shared_instruction_head(self):
        """Different agents' classifier prompts share the 28-31K instructions and differ in a few thousand tokens of
        transcript: reusing the head is worth more than the transcript it overwrites."""
        slots = HybridCacheSlots(4)
        a = CLS_ROOT + toks(50, 3000) + toks(90, 300)
        b = CLS_ROOT + toks(51, 2000) + toks(90, 300)
        self.assertEqual(slots.choose(a)[0], 0)
        self.assertEqual(slots.choose(b), (0, 28000, "joins"))
        self.assertEqual(slots.choose(a + toks(52, 400))[:2], (0, 28000))   # a's next round: the head again

    def test_two_users_long_conversations_do_not_join_on_a_shared_head(self):
        slots = HybridCacheSlots(4)
        a, b = MAIN_HEAD + toks(10, 15000), MAIN_HEAD + toks(11, 15000)    # 25K each, 10K shared: loss 15K > 10K
        self.assertEqual(slots.choose(a)[0], 0)
        self.assertEqual(slots.choose(b)[0], 1)
        for k in range(6):
            a, b = a + toks(20 + k, 1000), b + toks(30 + k, 1000)
            self.assertEqual(slots.choose(a)[::2], (0, "continues"))
            self.assertEqual(slots.choose(b)[::2], (1, "continues"))

    def test_short_conversations_on_a_shared_head_join_while_the_loss_is_under_the_head(self):
        """The bounded case: two conversations each under twice the head share a slot (each re-reads at most its own
        part past the head, less than the head reused) until one grows past twice the head."""
        slots = HybridCacheSlots(4)
        a = MAIN_HEAD + toks(10, 2000)
        slots.choose(a)
        self.assertEqual(slots.choose(MAIN_HEAD + toks(11, 2000)), (0, 10000, "joins"))
        self.assertEqual(slots.choose(MAIN_HEAD + toks(12, 25000))[0], 0)    # joins (loss 2,000 <= 10,000)
        self.assertEqual(slots.choose(MAIN_HEAD + toks(13, 2000))[::2], (1, "free"))   # 35K there: loss 25K > 10K

    def test_the_longest_shared_prefix_wins(self):
        slots = HybridCacheSlots(4)
        slots.choose(CLS_ROOT + toks(50, 1000))
        slots.choose(SUB_HEAD + toks(60, 1000))
        slots.record(2, CLS_ROOT + toks(50, 1000) + toks(51, 1000))
        self.assertEqual(slots.choose(CLS_ROOT + toks(50, 1000) + toks(51, 1000) + toks(52, 10))[:2], (2, 30000))

    def test_a_new_subagent_joins_its_finished_sibling_not_the_idle_main(self):
        """I1: main idles through a subagent run; the next subagent of the same type shares its 12K head with the
        finished one (loss 5K < 12K) and takes that slot, not main's by age."""
        slots = HybridCacheSlots(4)
        main = MAIN_HEAD + toks(10, 90000)
        sub1 = SUB_HEAD + toks(30, 5000)
        for ids in (main, toks(70, 300), CLS_ROOT + toks(51, 500), sub1):
            slots.choose(ids)
        self.assertEqual(slots.choose(SUB_HEAD + toks(40, 2000))[::2], (3, "joins"))
        self.assertEqual(slots.choose(main + toks(99, 1500))[::2], (0, "continues"))

    def test_a_side_request_evicts_a_slot_no_bigger_than_itself(self):
        slots = HybridCacheSlots(4)
        main = MAIN_HEAD + toks(10, 90000)
        for ids in (main, toks(70, 300), CLS_ROOT + toks(51, 500), SUB_HEAD + toks(30, 5000)):
            slots.choose(ids)
        for k in range(4):                          # main is the least recently used; the 300-token title's slot goes
            self.assertEqual(slots.choose(toks(80 + k, 400))[::2], (1, "evicts"), k)

    def test_with_no_slot_that_small_the_least_recently_used_goes(self):
        """The w2a shape: two finished long sessions and two live ones; a new session (no slot within its own size)
        takes the least recently used slot - a finished one - not the live conversation holding the fewest tokens."""
        slots = HybridCacheSlots(4)
        finished_a, finished_b = MAIN_HEAD + toks(10, 90000), MAIN_HEAD + toks(11, 50000)
        live_a, live_b = MAIN_HEAD + toks(12, 15000), MAIN_HEAD + toks(13, 40000)
        for ids in (finished_a, finished_b, live_a, live_b):
            slots.choose(ids)
        live_a, live_b = live_a + [1], live_b + [1]
        slots.choose(live_a)
        slots.choose(live_b)
        self.assertEqual(slots.choose(MAIN_HEAD + toks(14, 2000))[::2], (0, "evicts"))   # 12K: no loss <= 12K
        self.assertEqual(slots.choose(finished_b + [2])[::2], (1, "continues"))      # the finished one kept

    def test_a_live_conversation_within_the_newcomers_size_can_go(self):
        """The rule's bound, not liveness: a slot whose loss is at most the newcomer's size is a candidate even if
        it is live; the worst extra read is then the newcomer's own size."""
        slots = HybridCacheSlots(2)
        slots.choose(MAIN_HEAD + toks(10, 90000))
        slots.choose(MAIN_HEAD + toks(12, 12000))
        self.assertEqual(slots.choose(MAIN_HEAD + toks(14, 12000))[::2], (1, "evicts"))

    def test_a_pinned_slot_is_neither_joined_nor_evicted_while_another_can_go(self):
        slots = HybridCacheSlots(2)
        pinned = MAIN_HEAD + toks(10, 3000)
        slots.record(0, pinned)
        self.assertEqual(slots.choose(MAIN_HEAD + toks(11, 3000))[::2], (1, "free"))
        self.assertEqual(slots.choose(MAIN_HEAD + toks(12, 3000))[::2], (1, "joins"))  # not the pinned slot 0
        self.assertEqual(slots.choose(toks(70, 300))[::2], (1, "evicts"))
        self.assertEqual(slots.choose(pinned + toks(13, 100))[::2], (0, "continues"))  # its own continuation may land
        slots.record(1, toks(71, 300))
        self.assertEqual(slots.choose(toks(72, 300))[::2], (0, "evicts"))   # every slot pinned: by age

    def test_shared_and_pick_agree_with_choose(self):
        slots = HybridCacheSlots(4)
        head = list(range(600))
        slots.pick(head + [1, 2, 3])
        self.assertEqual(slots.shared(head + [1, 9, 9, 9]), 601)
        self.assertIsNone(slots.shared([7] * 600))


class WithSlotHybrid(unittest.TestCase):
    def service(self):
        tok = ByteTokenizer()
        eng = MockEngine(tok, "ok", max_context=16384)
        eng.info = {"cache_slots": 4}
        return Service(eng, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))

    def test_selected_by_the_variable_with_ckpt_at_and_a_debug_line(self):
        with mock.patch.dict(os.environ, {"STRATA_CACHE_SLOT_RULE": "hybrid", "STRATA_DEBUG": "1"}), \
                mock.patch("builtins.print") as p:
            svc = self.service()
            a = svc.with_slot({}, CLS_ROOT + toks(50, 3000))
            b = svc.with_slot({}, CLS_ROOT + toks(51, 2000))
            c = svc.with_slot({"strata_cache_slot": 3}, toks(70, 300))
        self.assertIsInstance(svc.slots, HybridCacheSlots)
        self.assertNotIn("_ckpt_at", a)
        self.assertEqual((b["strata_cache_slot"], b["_ckpt_at"]), (0, 28000))
        self.assertEqual(c, {"strata_cache_slot": 3})
        self.assertTrue(svc.slots.table[3]["pinned"])
        lines = [x.args[0] for x in p.call_args_list]
        self.assertIn("slot 0 (joins): 30000 tokens, shares 28000 of its last 31000", lines[1])


if __name__ == "__main__":
    unittest.main()
