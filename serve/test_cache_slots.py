"""serve/test_cache_slots.py - the server's choice of the engine's conversation-cache slot (#321), no engine.

    python -m pytest serve/test_cache_slots.py -q

Two users of one server whose prompts share the first 512 tokens (the same harness: system prompt and tools) share
one slot under the family rule and re-read each other's whole conversation on every alternated turn (w2a: 452/419 s
against 270/283 s with each user pinned to a slot).  STRATA_CACHE_SLOT_RULE picks the rule: unset or "family" the
family rule (CacheSlots, the default, as before #321), "identity" the family per sender (IdentityCacheSlots),
"continuation" the slot the prompt continues, else a free one, else by age (ContinuationCacheSlots).
STRATA_SLOT_TRACE records the request stream for tools/cacheslots_replay.py.
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import (ByteTokenizer, CacheSlots, ContinuationCacheSlots, IdentityCacheSlots,  # noqa: E402
                          MockEngine, Service, serve)

ROOT = Path(__file__).resolve().parents[1]
HEAD = list(range(600))                       # two users of one harness: the same first 600 tokens


def body(user, n):
    return [user * 1_000_000 + i for i in range(n)]


def turn(user, t, step=1200):
    """User `user`'s prompt at turn t: the shared head, then its own conversation, growing by `step` per turn.  Its own
    part is over ContinuationCacheSlots.TAIL from the first turn; a shorter one is test_a_short_first_turn_can_continue_another_users_slot."""
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
        slots = ContinuationCacheSlots(4)
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
        slots = ContinuationCacheSlots(4)
        got = [slots.choose(turn(1, t))[0] for t in range(12)]
        self.assertEqual(got, [0] * 12)

    def test_side_requests_never_evict_a_conversation_while_a_short_slot_exists(self):
        slots = ContinuationCacheSlots(4)
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

    def test_with_three_users_a_side_request_takes_a_side_slot(self):
        slots = ContinuationCacheSlots(4)
        for user, t in ((1, 0), (2, 1), (3, 2)):    # users of 1,800 / 3,000 / 4,200 tokens: all over SIDE_MAX
            slots.choose(turn(user, t))
        lens = {s: len(slots.table[s]["ids"]) for s in range(3)}
        self.assertEqual(lens, {0: 1800, 1: 3000, 2: 4200})
        self.assertTrue(all(n >= ContinuationCacheSlots.SIDE_MAX for n in lens.values()))
        slot, shared, how = slots.choose(side(0))   # the free slot first
        self.assertEqual((slot, shared, how), (3, None, "free"))
        for k in (1, 2, 3):                         # then the short slot, though user 1's is the least recently used
            self.assertEqual(slots.choose(side(k))[::2], (3, "evicts"), k)
        slot, _, how = slots.choose(turn(4, 0))     # a long new prompt: the least recently used slot of all (user 1)
        self.assertEqual((slot, how), (0, "evicts"))
        self.assertEqual(slots.choose(side(9))[::2], (3, "evicts"))   # the side slot again, not a user's

    def test_a_short_prompt_with_no_short_slot_takes_the_least_recently_used(self):
        slots = ContinuationCacheSlots(2)
        slots.choose(turn(1, 0))
        slots.choose(turn(2, 0))
        slots.choose(turn(1, 1))                    # user 1 is the more recent
        self.assertEqual(slots.choose(side(0))[::2], (1, "evicts"))

    def test_a_finished_long_session_goes_before_a_live_short_one(self):
        """The w2a shape that fewest-tokens eviction got wrong (#321 replay): two users each finished a long session
        and started a shorter one; a new session must evict a finished session's slot (the least recently used), not
        a live conversation that merely holds fewer tokens."""
        slots = ContinuationCacheSlots(4)
        finished_a, finished_b = turn(1, 20), turn(2, 15)           # 25,800 / 19,800 tokens, nothing continues them
        live_a, live_b = HEAD + body(3, 3000), HEAD + body(4, 3000)  # 3,600 each
        for ids in (finished_a, finished_b, live_a, live_b):
            slots.choose(ids)
        for t in range(3):                                          # the live ones keep going
            live_a, live_b = live_a + body(5, 100 * (t + 1)), live_b + body(6, 100 * (t + 1))
            self.assertEqual(slots.choose(live_a)[::2], (2, "continues"))
            self.assertEqual(slots.choose(live_b)[::2], (3, "continues"))
        new_b = HEAD + body(7, 2000)
        self.assertEqual(slots.choose(new_b)[::2], (0, "evicts"))   # A's finished session, not live A (fewest)
        self.assertEqual(slots.choose(live_a + [1])[::2], (2, "continues"))
        self.assertEqual(slots.choose(HEAD + body(8, 2000))[::2], (1, "evicts"))   # then B's finished one

    def test_ties_go_to_the_least_recently_used(self):
        slots = ContinuationCacheSlots(2)
        slots.choose(side(0))                       # slot 0
        slots.choose(side(1))                       # slot 1
        slots.choose(side(0))                       # an exact repeat under 512 tokens is no continuation: evicts 0
        self.assertEqual(slots.choose(side(2))[0], 1)   # 0 and 1 hold 400 each; 1 is the older

    def test_a_rewind_mid_way_takes_another_slot(self):
        slots = ContinuationCacheSlots(4)
        for t in range(10):
            slots.choose(turn(1, t))                # 3,600 tokens in slot 0
        rewound = turn(1, 9)[:2000] + [7] * 100     # compaction / rewind: parts at 2,000 of 3,600
        slot, shared, how = slots.choose(rewound)
        self.assertEqual((slot, shared, how), (1, None, "free"))
        self.assertEqual(slots.choose(rewound + [8] * 50)[:2], (1, 2100))   # it grows on in its new slot

    def test_a_tail_up_to_TAIL_is_still_a_continuation(self):
        """A transcript growing inside one message (the classifier's): the last prompt's own tail after the shared
        point (its closing instructions) is not in the next prompt."""
        slots = ContinuationCacheSlots(4)
        root, suffix = list(range(2000)), [5_000_000 + i for i in range(ContinuationCacheSlots.TAIL)]
        slots.choose(root + body(1, 3000) + suffix)
        slot, shared, how = slots.choose(root + body(1, 4000) + suffix)
        self.assertEqual((slot, shared, how), (0, 5000, "continues"))
        long_suffix = suffix + [6_000_000]          # one token over TAIL: not a continuation
        slots2 = ContinuationCacheSlots(4)
        slots2.choose(root + body(1, 3000) + long_suffix)
        self.assertEqual(slots2.choose(root + body(1, 4000) + long_suffix), (1, None, "free"))

    def test_the_longest_shared_prefix_wins_among_continuations(self):
        slots = ContinuationCacheSlots(4)
        l0 = HEAD + body(1, 1000)
        l1 = l0 + body(2, 500)
        slots.record(0, l0)
        slots.record(1, l1)
        self.assertEqual(slots.choose(l1 + body(3, 100)), (1, 2100, "continues"))   # continues both; 1 is longer

    def test_a_short_first_turn_can_continue_another_users_slot(self):
        """TAIL is also a window for a false continuation: B's first prompt continues A's first when A's own part is
        under TAIL tokens.  Bounded: they part once either conversation grows past TAIL (here at A's second turn)."""
        slots = ContinuationCacheSlots(4)
        self.assertEqual(slots.choose(HEAD + body(1, 100))[0], 0)
        self.assertEqual(slots.choose(HEAD + body(2, 100)), (0, 600, "continues"))
        self.assertEqual(slots.choose(HEAD + body(1, 2000))[0], 0)   # A again: continues B's 700 (shares 600)
        self.assertEqual(slots.choose(HEAD + body(2, 2000))[2], "free")   # B parts: A's last is 2,600 long

    def test_under_PREFIX_tokens_is_never_a_continuation(self):
        slots = ContinuationCacheSlots(4)
        slots.choose(list(range(300)))
        self.assertEqual(slots.choose(list(range(300)))[2], "free")

    def test_a_pinned_slot_is_recorded(self):
        slots = ContinuationCacheSlots(4)
        slots.record(0, turn(1, 0))
        self.assertEqual(slots.choose(turn(2, 0))[0], 1)
        self.assertEqual(slots.choose(turn(1, 1)), (0, 1800, "continues"))

    def test_ids_of_any_sequence_type(self):
        """_shared_len compares list slices, and a tuple slice never equals a list slice: ids are made a list first."""
        slots = ContinuationCacheSlots(4)
        slots.record(0, tuple(turn(1, 3)))
        self.assertEqual(slots.choose(tuple(turn(1, 4))), (0, len(turn(1, 3)), "continues"))
        self.assertEqual(slots.shared(tuple(turn(1, 5))), len(turn(1, 4)))

    def test_shared_and_pick_answer_like_choose(self):
        slots = ContinuationCacheSlots(4)
        head = list(range(600))
        slots.pick(head + [1, 2, 3])
        self.assertEqual(slots.shared(head + [1, 9, 9, 9]), 601)
        self.assertIsNone(slots.shared([7] * 600))


class WithSlot(unittest.TestCase):
    """Service.with_slot under STRATA_CACHE_SLOT_RULE=continuation: the slot and _ckpt_at it puts in the request,
    and the pinned requests it records."""

    def setUp(self):
        patcher = mock.patch.dict(os.environ, {"STRATA_CACHE_SLOT_RULE": "continuation"}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)

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
                self.assertLess(len(last) - shared, ContinuationCacheSlots.TAIL)


class FamilyRule(unittest.TestCase):
    """The family rule (CacheSlots), the default: unset STRATA_CACHE_SLOT_RULE or "family" is the rule before #321,
    byte for byte (the golden below was taken from the class before any #321 change, on the same inputs)."""
    GOLDEN = [(0, None), (1, None), (0, 600), (2, None), (0, 600), (3, None), (0, 600), (1, None), (0, 600),
              (2, None), (0, 600), (3, None), (0, 600), (0, 600), (0, 600), (0, 600)]

    @staticmethod
    def inputs(step=300):
        seqs = []
        for t in range(3):
            seqs += [turn(1, t, step), side(2 * t), turn(2, t, step), side(2 * t + 1)]
        return seqs + [turn(1, 3, step)[:700] + [5] * 50, turn(3, 0, step), turn(1, 4, step), turn(2, 3, step)]

    def run_rule(self, env, who=lambda i: None):
        with mock.patch.dict(os.environ, env):
            if not env:
                os.environ.pop("STRATA_CACHE_SLOT_RULE", None)
            svc = service()
            got = []
            for i, ids in enumerate(self.inputs()):
                if i == 5:                                          # a pinned request: the family rule ignores it
                    self.assertEqual(svc.with_slot({"strata_cache_slot": 2}, turn(4, 0), who(i)),
                                     {"strata_cache_slot": 2})
                r = svc.with_slot({}, ids, who(i))
                got.append((r["strata_cache_slot"], r.get("_ckpt_at")))
        return got

    def test_the_default_is_the_family_rule(self):
        self.assertEqual(self.run_rule({}), self.GOLDEN)
        self.assertEqual(self.run_rule({"STRATA_CACHE_SLOT_RULE": "family"}), self.GOLDEN)
        self.assertEqual(self.run_rule({"STRATA_CACHE_SLOT_RULE": "no such rule"}), self.GOLDEN)

    def test_the_family_rule_ignores_who(self):
        who = lambda i: {"key": "owner", "device": f"addr:10.0.0.{i % 3}", "addr": f"10.0.0.{i % 3}"}  # noqa: E731
        self.assertEqual(self.run_rule({}, who), self.GOLDEN)

    def test_the_continuation_rule_differs_on_the_same_inputs(self):
        with mock.patch.dict(os.environ, {"STRATA_CACHE_SLOT_RULE": "continuation"}):
            svc = service()
            got = [svc.with_slot({}, ids)["strata_cache_slot"] for ids in self.inputs(step=1200)]
        self.assertNotEqual(got, [s for s, _ in self.GOLDEN])
        self.assertEqual(got[0], got[4])                            # user 1 turns 0 -> 1 keep a slot
        self.assertNotEqual(got[0], got[2])                         # users 1 and 2 never share


class IdentityRule(unittest.TestCase):
    """STRATA_CACHE_SLOT_RULE=identity: the family rule with the family keyed by (key name, device) as well."""
    ENV = {"STRATA_CACHE_SLOT_RULE": "identity"}

    @staticmethod
    def who(user):
        return {"key": "owner", "device": f"addr:192.168.1.{user}", "addr": f"192.168.1.{user}"}

    def test_one_identity_is_exactly_the_family_rule(self):
        rule = FamilyRule("test_the_default_is_the_family_rule")
        self.assertEqual(rule.run_rule(self.ENV, lambda i: self.who(1)), FamilyRule.GOLDEN)
        self.assertEqual(rule.run_rule(self.ENV), FamilyRule.GOLDEN)           # the server's own requests: None

    def test_two_identities_with_one_head_never_share_a_free_slot(self):
        with mock.patch.dict(os.environ, self.ENV):
            svc = service()
            got = {1: set(), 2: set()}
            for t in range(10):
                for user in (1, 2):
                    r = svc.with_slot({}, turn(user, t, 300), self.who(user))
                    got[user].add(r["strata_cache_slot"])
                    if t:
                        self.assertEqual(r["_ckpt_at"], len(turn(user, t - 1, 300)))
        self.assertEqual((got[1], got[2]), ({0}, {1}))

    def test_with_no_free_slot_the_least_recently_used_family_goes(self):
        slots = IdentityCacheSlots(2)
        a, b = ("owner", "addr:1"), ("owner", "addr:2")
        self.assertEqual(slots.choose(turn(1, 0), who=a)[0], 0)
        self.assertEqual(slots.choose(turn(2, 0), who=b)[0], 1)
        self.assertEqual(slots.choose(turn(1, 1), who=a)[:2], (0, 1800))
        self.assertEqual(slots.choose(side(0), who=b)[:2], (1, None))      # b's family is the least recently used
        self.assertEqual(slots.choose(turn(2, 1), who=b)[:2], (0, None))   # b's turn: gone; now a's is the oldest

    def test_the_key_name_and_the_device_both_part_identities(self):
        slots = IdentityCacheSlots(4)
        same_head = turn(1, 0)
        got = [slots.choose(same_head, who=w)[0] for w in
               (("alice", "addr:1"), ("bob", "addr:1"), ("alice", "dev-aaaaaaaa"), ("alice", "addr:1"))]
        self.assertEqual(got, [0, 1, 2, 0])


class SlotTrace(unittest.TestCase):
    """STRATA_SLOT_TRACE=<path>: one JSON line per request that reaches with_slot, for tools/cacheslots_replay.py."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, True)

    @staticmethod
    def lines(path):
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            return [json.loads(x) for x in f if x.strip()]

    def test_off_by_default(self):
        with mock.patch.dict(os.environ, {}):
            os.environ.pop("STRATA_SLOT_TRACE", None)
            with mock.patch("serve.server.slot_trace") as w:
                service().with_slot({}, turn(1, 0))
        w.assert_not_called()

    def test_a_line_per_request(self):
        for name in ("t.jsonl", "t.jsonl.gz"):
            path = self.dir / name
            with mock.patch.dict(os.environ, {"STRATA_SLOT_TRACE": str(path)}):
                svc = service()
                who = {"key": "alice", "device": "addr:10.0.0.7", "addr": "10.0.0.7", "secret": "sk-never"}
                svc.with_slot({"max_tokens": 64}, turn(1, 0), who, "anthropic")
                svc.with_slot({"max_completion_tokens": 9, "strata_cache_slot": 2}, turn(1, 1), None, "openai")
            one, two = self.lines(path)
            self.assertEqual(one["who"], {"key": "alice", "device": "addr:10.0.0.7", "addr": "10.0.0.7"})
            self.assertEqual((one["route"], one["max_tokens"], one["pinned"], one["cache_slots"], one["n"]),
                             ("anthropic", 64, None, 4, 1800))
            self.assertEqual(one["ids"], turn(1, 0))
            self.assertEqual((two["who"], two["route"], two["max_tokens"], two["pinned"]), (None, "openai", 9, 2))
            self.assertGreater(two["t"], 1.7e9)

    def test_both_routes_send_who(self):
        """The HTTP path: the openai and anthropic routes pass the handler's _slot_who (no keys file: "owner", the
        socket peer's address, the device header when sent) and their route name."""
        path = self.dir / "http.jsonl"
        tok = ByteTokenizer()
        eng = MockEngine(tok, "</think>\n\nok", max_context=16384)
        eng.info = {"cache_slots": 4}
        svc = Service(eng, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        msgs = [{"role": "user", "content": "hi"}]
        with mock.patch.dict(os.environ, {"STRATA_SLOT_TRACE": str(path)}):
            for route, body, headers in (
                    ("/v1/chat/completions", {"model": "m", "messages": msgs, "max_tokens": 8}, {}),
                    ("/v1/messages", {"model": "m", "messages": msgs, "max_tokens": 8},
                     {"X-Strata-Device": "dev-12345678"})):
                req = urllib.request.Request(base + route, data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json", **headers})
                with urllib.request.urlopen(req, timeout=30) as r:
                    r.read()
        got = [(x["route"], x["who"]) for x in self.lines(path)]
        self.assertEqual(got, [("openai", {"key": "owner", "device": "addr:127.0.0.1", "addr": "127.0.0.1"}),
                               ("anthropic", {"key": "owner", "device": "dev-12345678", "addr": "127.0.0.1"})])


if __name__ == "__main__":
    unittest.main()
