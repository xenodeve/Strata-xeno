"""serve/test_cache_slots.py - the server's choice of the engine's conversation-cache slot (#321), no engine.

    python -m pytest serve/test_cache_slots.py -q

Two users of one server whose prompts share the first 512 tokens (the same harness: system prompt and tools) shared
one slot under the family rule and re-read each other's whole conversation on every alternated turn (w2a: 452/419 s
against 270/283 s with each user pinned to a slot); a side request with a new head took the least recently used slot,
which could be a user's conversation (w2s: re-reads of 45-121K tokens).  The slot rule (default) sends a prompt to the
slot whose last prompt it continues, else a free slot, else evicts the slot holding the fewest tokens.
STRATA_CACHE_SLOT_RULE=family keeps the old rule for the A/B.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, CacheSlots, MockEngine, Service  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HEAD = list(range(600))                       # two users of one harness: the same first 600 tokens


def body(user, n):
    return [user * 1_000_000 + i for i in range(n)]


def turn(user, t, step=1200):
    """User `user`'s prompt at turn t: the shared head, then its own conversation, growing by `step` per turn.  Its own
    part is over CacheSlots.TAIL from the first turn; a shorter one is test_a_short_first_turn_can_continue_another_users_slot."""
    return HEAD + body(user, step * (t + 1))


def side(k, n=400):
    """A side request (auto-mode classifier, title): its own head every time, a few hundred tokens."""
    return [9_000_000 + k * 1000 + i for i in range(n)]


def service(slots=4):
    tok = ByteTokenizer()
    eng = MockEngine(tok, "ok", max_context=16384)
    eng.info = {"cache_slots": slots}
    return Service(eng, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))


class SlotRule(unittest.TestCase):
    """The default rule, on CacheSlots directly: choose(ids) -> (slot, shared, how)."""

    def setUp(self):
        patcher = mock.patch.dict(os.environ, {}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("STRATA_CACHE_SLOT_RULE", None)

    def test_two_users_with_one_head_keep_their_own_slot(self):
        slots = CacheSlots(4)
        seen = {1: set(), 2: set()}
        for t in range(10):
            for user in (1, 2):
                slot, _, how = slots.choose(turn(user, t))
                seen[user].add(slot)
                self.assertEqual(how, "free" if t == 0 else "continues", (user, t))
        self.assertEqual(len(seen[1]), 1)
        self.assertEqual(len(seen[2]), 1)
        self.assertNotEqual(seen[1], seen[2])

    def test_a_growing_conversation_stays_in_one_slot(self):
        slots = CacheSlots(4)
        got = [slots.choose(turn(1, t))[0] for t in range(12)]
        self.assertEqual(got, [0] * 12)

    def test_side_requests_never_evict_a_conversation_while_a_short_slot_exists(self):
        slots = CacheSlots(4)
        main = {}
        k = 0
        for t in range(8):
            for user in (1, 2):
                slot, _, _ = slots.choose(turn(user, t))
                self.assertEqual(main.setdefault(user, slot), slot, (user, t))
                s, _, how = slots.choose(side(k))
                k += 1
                self.assertNotIn(s, main.values())
                self.assertIn(how, ("free", "evicts"))
        self.assertEqual(sorted(main.values()), [0, 2])             # user 1, its side request, user 2

    def test_with_three_users_the_fewest_token_slot_goes(self):
        slots = CacheSlots(4)
        for user, t in ((1, 0), (2, 1), (3, 2)):    # users of 1,800 / 3,000 / 4,200 tokens
            slots.choose(turn(user, t))
        lens = {s: len(slots.table[s]["ids"]) for s in range(3)}
        self.assertEqual(lens, {0: 1800, 1: 3000, 2: 4200})
        slot, shared, how = slots.choose(side(0))   # the free slot first
        self.assertEqual((slot, shared, how), (3, None, "free"))
        slot, _, how = slots.choose(side(1))        # then the fewest tokens: the other side request (400)
        self.assertEqual((slot, how), (3, "evicts"))
        slot, _, how = slots.choose(side(2, n=2000))  # 2,000 tokens: evicts the 400-token slot again
        self.assertEqual((slot, how), (3, "evicts"))
        slot, _, how = slots.choose(side(5))        # now the 1,800-token user (fewest), not the 2,000-token side
        self.assertEqual((slot, how), (0, "evicts"))

    def test_ties_go_to_the_least_recently_used(self):
        slots = CacheSlots(2)
        slots.choose(side(0))                       # slot 0
        slots.choose(side(1))                       # slot 1
        slots.choose(side(0))                       # an exact repeat under 512 tokens is no continuation: evicts 0
        self.assertEqual(slots.choose(side(2))[0], 1)   # 0 and 1 hold 400 each; 1 is the older

    def test_a_rewind_mid_way_takes_another_slot(self):
        slots = CacheSlots(4)
        for t in range(10):
            slots.choose(turn(1, t))                # 3,600 tokens in slot 0
        rewound = turn(1, 9)[:2000] + [7] * 100     # compaction / rewind: parts at 2,000 of 3,600
        slot, shared, how = slots.choose(rewound)
        self.assertEqual((slot, shared, how), (1, None, "free"))
        self.assertEqual(slots.choose(rewound + [8] * 50)[:2], (1, 2100))   # it grows on in its new slot

    def test_a_tail_up_to_TAIL_is_still_a_continuation(self):
        """A transcript growing inside one message (the classifier's): the last prompt's own tail after the shared
        point (its closing instructions) is not in the next prompt."""
        slots = CacheSlots(4)
        root, suffix = list(range(2000)), [5_000_000 + i for i in range(CacheSlots.TAIL)]
        slots.choose(root + body(1, 3000) + suffix)
        slot, shared, how = slots.choose(root + body(1, 4000) + suffix)
        self.assertEqual((slot, shared, how), (0, 5000, "continues"))
        long_suffix = suffix + [6_000_000]          # one token over TAIL: not a continuation
        slots2 = CacheSlots(4)
        slots2.choose(root + body(1, 3000) + long_suffix)
        self.assertEqual(slots2.choose(root + body(1, 4000) + long_suffix), (1, None, "free"))

    def test_the_longest_shared_prefix_wins_among_continuations(self):
        slots = CacheSlots(4)
        l0 = HEAD + body(1, 1000)
        l1 = l0 + body(2, 500)
        slots.record(0, l0)
        slots.record(1, l1)
        self.assertEqual(slots.choose(l1 + body(3, 100)), (1, 2100, "continues"))   # continues both; 1 is longer

    def test_a_short_first_turn_can_continue_another_users_slot(self):
        """TAIL is also a window for a false continuation: B's first prompt continues A's first when A's own part is
        under TAIL tokens.  Bounded: they part once either conversation grows past TAIL (here at A's second turn)."""
        slots = CacheSlots(4)
        self.assertEqual(slots.choose(HEAD + body(1, 100))[0], 0)
        self.assertEqual(slots.choose(HEAD + body(2, 100)), (0, 600, "continues"))
        self.assertEqual(slots.choose(HEAD + body(1, 2000))[0], 0)   # A again: continues B's 700 (shares 600)
        self.assertEqual(slots.choose(HEAD + body(2, 2000))[2], "free")   # B parts: A's last is 2,600 long

    def test_under_PREFIX_tokens_is_never_a_continuation(self):
        slots = CacheSlots(4)
        slots.choose(list(range(300)))
        self.assertEqual(slots.choose(list(range(300)))[2], "free")

    def test_a_pinned_slot_is_recorded(self):
        slots = CacheSlots(4)
        slots.record(0, turn(1, 0))
        self.assertEqual(slots.choose(turn(2, 0))[0], 1)
        self.assertEqual(slots.choose(turn(1, 1)), (0, 1800, "continues"))

    def test_ids_of_any_sequence_type(self):
        """_shared_len compares list slices, and a tuple slice never equals a list slice: ids are made a list first."""
        slots = CacheSlots(4)
        slots.record(0, tuple(turn(1, 3)))
        self.assertEqual(slots.choose(tuple(turn(1, 4))), (0, len(turn(1, 3)), "continues"))
        self.assertEqual(slots.shared(tuple(turn(1, 5))), len(turn(1, 4)))

    def test_shared_and_pick_answer_like_choose(self):
        slots = CacheSlots(4)
        head = list(range(600))
        slots.pick(head + [1, 2, 3])
        self.assertEqual(slots.shared(head + [1, 9, 9, 9]), 601)
        self.assertIsNone(slots.shared([7] * 600))


class WithSlot(unittest.TestCase):
    """Service.with_slot: the slot and _ckpt_at it puts in the request, and the pinned requests it records."""

    def setUp(self):
        patcher = mock.patch.dict(os.environ, {}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("STRATA_CACHE_SLOT_RULE", None)

    def test_ckpt_at_is_the_shared_prefix_of_a_continuation_only(self):
        svc = service()
        root, suffix = list(range(2000)), [5_000_000 + i for i in range(300)]
        first = svc.with_slot({}, root + body(1, 3000) + suffix)
        self.assertNotIn("_ckpt_at", first)                         # nothing to compare with
        nxt = svc.with_slot({}, root + body(1, 4000) + suffix)
        self.assertEqual((nxt["strata_cache_slot"], nxt["_ckpt_at"]), (first["strata_cache_slot"], 5000))
        other = svc.with_slot({}, root + body(2, 3000))             # shares the 2,000 root only: a new slot
        self.assertNotEqual(other["strata_cache_slot"], first["strata_cache_slot"])
        self.assertNotIn("_ckpt_at", other)
        rewound = svc.with_slot({}, root + body(1, 2500) + [7])     # a rewind: no checkpoint at the parting
        self.assertNotIn("_ckpt_at", rewound)

    def test_a_client_pinned_slot_is_recorded_and_left_as_sent(self):
        svc = service()
        req = {"strata_cache_slot": 0}
        out = svc.with_slot(req, turn(1, 0))
        self.assertEqual(out, {"strata_cache_slot": 0})
        b = svc.with_slot({}, turn(2, 0))
        self.assertEqual(b["strata_cache_slot"], 1)                 # slot 0 is taken: a free one
        c = svc.with_slot({}, turn(3, 0))
        self.assertEqual(c["strata_cache_slot"], 2)
        again = svc.with_slot({}, turn(1, 1))                       # the pinned user, unpinned now: its slot
        self.assertEqual((again["strata_cache_slot"], again["_ckpt_at"]), (0, 1800))

    def test_a_bad_pinned_slot_is_left_to_the_validation(self):
        svc = service(2)
        for bad in (3, -1, "1", True, 1.5, None):
            self.assertEqual(svc.with_slot({"strata_cache_slot": bad}, turn(1, 0)), {"strata_cache_slot": bad})
        self.assertEqual(svc.with_slot({}, turn(2, 0))["strata_cache_slot"], 0)   # nothing was recorded

    def test_debug_line_says_how(self):
        svc = service()
        with mock.patch.dict(os.environ, {"STRATA_DEBUG": "1"}), mock.patch("builtins.print") as p:
            svc.with_slot({}, turn(1, 0))
            svc.with_slot({}, turn(1, 1))
            svc.with_slot({}, turn(2, 0))
        lines = [c.args[0] for c in p.call_args_list]
        self.assertIn("slot 0 (free): 1800 tokens", lines[0])
        self.assertIn("slot 0 (continues): 3000 tokens, shares 1800", lines[1])
        self.assertIn("slot 1 (free): 1800 tokens", lines[2])
        self.assertIn("nearest: slot 0 shares 600, tail 2400", lines[2])

    def test_consecutive_turns_through_the_template_part_within_TAIL(self):
        """How the server renders a previous turn: the template keeps every turn's thinking (preserve_thinking is
        never set false in serve/), so the next prompt repeats the last one up to its generation prompt; what can
        differ is that opening's own few tokens (thinking returned, dropped, off; a tool turn)."""
        svc = service()
        sys_ = "You are an agent. " * 40
        tools = [{"type": "function", "function": {"name": "Bash", "description": "run",
                                                   "parameters": {"type": "object", "properties": {}}}}]
        user = [{"role": "system", "content": sys_}, {"role": "user", "content": "fix the bug"}]
        cases = {
            "thinking returned": ({}, {"role": "assistant", "content": "done", "reasoning_content": "I think"}),
            "thinking dropped": ({}, {"role": "assistant", "content": "done"}),
            "thinking off": ({"enable_thinking": False}, {"role": "assistant", "content": "done"}),
            "tool turn": ({}, {"role": "assistant", "content": "", "reasoning_content": "run it",
                               "tool_calls": [{"type": "function", "id": "c1",
                                               "function": {"name": "Bash", "arguments": {}}}]}),
        }
        for name, (kw, reply) in cases.items():
            with self.subTest(name):
                last = svc.encode_prompt(user, tools, dict(kw))
                follow = {"role": "tool", "tool_call_id": "c1", "content": "ok"} if reply.get("tool_calls") \
                    else {"role": "user", "content": "thanks, next"}
                nxt = svc.encode_prompt(user + [reply, follow], tools, dict(kw))
                shared = next((i for i in range(len(last)) if last[i] != nxt[i]), len(last))
                self.assertLessEqual(len(last) - shared, 8, name)
                self.assertLess(len(last) - shared, CacheSlots.TAIL)


class FamilyRule(unittest.TestCase):
    """STRATA_CACHE_SLOT_RULE=family: the rule before #321, byte for byte (the golden below was taken from the class
    before the change, on the same inputs)."""
    GOLDEN = [(0, None), (1, None), (0, 600), (2, None), (0, 600), (3, None), (0, 600), (1, None), (0, 600),
              (2, None), (0, 600), (3, None), (0, 600), (0, 600), (0, 600), (0, 600)]

    @staticmethod
    def inputs(step=300):
        seqs = []
        for t in range(3):
            seqs += [turn(1, t, step), side(2 * t), turn(2, t, step), side(2 * t + 1)]
        return seqs + [turn(1, 3, step)[:700] + [5] * 50, turn(3, 0, step), turn(1, 4, step), turn(2, 3, step)]

    def test_the_family_rule_reproduces_the_old_assignments(self):
        with mock.patch.dict(os.environ, {"STRATA_CACHE_SLOT_RULE": "family"}):
            svc = service()
            got = []
            for i, ids in enumerate(self.inputs()):
                if i == 5:                                          # a pinned request: the old rule ignores it
                    self.assertEqual(svc.with_slot({"strata_cache_slot": 2}, turn(4, 0)), {"strata_cache_slot": 2})
                r = svc.with_slot({}, ids)
                got.append((r["strata_cache_slot"], r.get("_ckpt_at")))
        self.assertEqual(got, self.GOLDEN)

    def test_the_default_rule_differs_on_the_same_inputs(self):
        os.environ.pop("STRATA_CACHE_SLOT_RULE", None)
        svc = service()
        got = [svc.with_slot({}, ids)["strata_cache_slot"] for ids in self.inputs(step=1200)]
        self.assertNotEqual(got, [s for s, _ in self.GOLDEN])
        self.assertEqual(got[0], got[4])                            # user 1 turns 0 -> 1 keep a slot
        self.assertNotEqual(got[0], got[2])                         # users 1 and 2 never share


if __name__ == "__main__":
    unittest.main()
