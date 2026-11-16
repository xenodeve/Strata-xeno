"""Tests for tools/workstation_bench.py, without an engine: config arms, the user split, record and log matching, the
two schedulers against a small threaded HTTP stub, and the process-tree stop on a harmless dummy tree.

    python tools/test_workstation_bench.py        (or: python -m pytest tools/test_workstation_bench.py)
"""
from __future__ import annotations

import copy
import http.server
import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import workstation_bench as WB  # noqa: E402


def base_config(profile: Path) -> dict:
    return {"parallel": 2, "host": "0.0.0.0", "api_key": "sk-test", "exe": "C:/x/strata.exe",
            "args": ["--pack", "p", "--expert-profile", str(profile), "--kv", "int8",
                     "--expert-profile-save", str(profile)],
            "log": "D:/real/strata.log", "port": 8091, "env": {"STRATA_A": "1", "STRATA_B": "2"}}


class ArmConfig(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.profile = self.dir / "real" / "profile.bin"
        self.profile.parent.mkdir()
        self.profile.write_bytes(b"PROFILE" * 100)
        self.base_path = self.dir / "real" / "base.json"
        self.base = base_config(self.profile)
        self.base_path.write_text(json.dumps(self.base), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def derive(self, spec, name="b0"):
        before = copy.deepcopy(self.base)
        cfg = WB.derive_config(self.base, WB.parse_arm(spec), self.dir / "work" / name, 8090)
        self.assertEqual(self.base, before, "the base dict changed")
        return cfg

    def test_parallel_removed_set_kept(self):
        self.assertNotIn("parallel", self.derive("P1:parallel=none"))
        self.assertEqual(self.derive("P4:parallel=4")["parallel"], 4)
        self.assertEqual(self.derive("K:")["parallel"], 2)

    def test_profile_save_redirected_to_a_copy_and_base_untouched(self):
        raw_before = self.base_path.read_bytes()
        prof_before = self.profile.read_bytes()
        cfg = self.derive("P2:parallel=2")
        args = cfg["args"]
        save = Path(args[args.index("--expert-profile-save") + 1])
        self.assertEqual(save.parent, self.dir / "work" / "b0")
        self.assertEqual(save.read_bytes(), prof_before)                     # a copy of the profile
        self.assertEqual(args[args.index("--expert-profile") + 1], str(self.profile))   # loads the same start
        save.write_bytes(b"changed by the engine")
        self.assertEqual(self.profile.read_bytes(), prof_before)
        self.assertEqual(self.base_path.read_bytes(), raw_before)
        p = WB.write_arm_config(cfg, self.dir / "work" / "b0")
        self.assertEqual(p.parent, self.dir / "work" / "b0")
        self.assertEqual(self.base_path.read_bytes(), raw_before)

    def test_profile_load_from_a_snapshot(self):
        snap = self.dir / "snap.bin"
        snap.write_bytes(b"SNAP")
        cfg = WB.derive_config(self.base, WB.parse_arm("P2:"), self.dir / "work" / "b1", 8090, str(snap))
        a = cfg["args"]
        self.assertEqual(a[a.index("--expert-profile") + 1], str(snap))
        self.assertEqual(Path(a[a.index("--expert-profile-save") + 1]).read_bytes(), self.profile.read_bytes())

    def test_log_history_host_key_port(self):
        cfg = self.derive("P2:parallel=2")
        bd = self.dir / "work" / "b0"
        self.assertEqual(Path(cfg["log"]), bd / "engine.log")
        self.assertEqual(Path(cfg["history"]["dir"]), bd / "history")
        self.assertEqual(cfg["host"], "127.0.0.1")
        self.assertNotIn("api_key", cfg)
        self.assertEqual(cfg["port"], 8090)

    def test_env_xargs_exe_and_boot_placeholder(self):
        cfg = self.derive("T:parallel=2,STRATA_TIMELINE={boot}/tl.json,STRATA_B=9,xargs=--foo;1,exe=C:/y/s.exe")
        bd = self.dir / "work" / "b0"
        self.assertEqual(cfg["env"], {"STRATA_A": "1", "STRATA_B": "9", "STRATA_TIMELINE": f"{bd}/tl.json"})
        self.assertEqual(cfg["args"][-2:], ["--foo", "1"])
        self.assertEqual(cfg["exe"], "C:/y/s.exe")
        env = WB.server_env(WB.parse_arm("T:STRATA_TIMELINE={boot}/tl.json"), bd, "1,0",
                            base_env={"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "PATH": "x"})
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1,0")
        self.assertNotIn("CUDA_DEVICE_ORDER", env)
        self.assertEqual(env["STRATA_TIMELINE"], f"{bd}/tl.json")

    def test_dump_routing_never_the_served_trace(self):
        self.base["args"] += ["--dump-routing", "D:/real/routing/served.bin"]
        cfg = self.derive("P2:")
        a = cfg["args"]
        self.assertEqual(Path(a[a.index("--dump-routing") + 1]).parent, self.dir / "work" / "b0")

    def test_parse_arm_errors(self):
        with self.assertRaises(ValueError):
            WB.parse_arm(":parallel=2")
        with self.assertRaises(ValueError):
            WB.parse_arm("X:parallel")


class Users(unittest.TestCase):
    def test_split_covers_every_turn_once_and_prefixes_differ(self):
        work = WB.workload()
        users = WB.split_users(work, "01/234")
        tags = [t for t, _ in users["A"]] + [t for t, _ in users["B"]]
        self.assertEqual(sorted(tags), sorted(t for t, _ in work))
        self.assertEqual(len(users["A"]), 10)
        self.assertEqual(len(users["B"]), 13)
        a0, b0 = users["A"][0][1], users["B"][0][1]
        self.assertEqual(a0[0], b0[0])                    # the shared system prompt
        self.assertNotEqual(a0[1]["content"], b0[1]["content"])   # each user's own task right after it
        # no turn of A starts with any turn of B's messages past the system prompt (distinct long prefixes)
        for _, ma in users["A"]:
            for _, mb in users["B"]:
                self.assertNotEqual(ma[1], mb[1])

    def test_consecutive_turns_of_a_user_share_their_prefix(self):
        users = WB.split_users(WB.workload(), "01/234")
        for u in "AB":
            for (t1, m1), (t2, m2) in zip(users[u], users[u][1:]):
                if t1.split(".")[0] == t2.split(".")[0]:
                    self.assertEqual(m2[:len(m1)], m1)

    def test_short_mode_extends_one_long_turn(self):
        items = WB.series_items("short", None, "01/234", 0, WB.workload())["solo"]
        self.assertEqual([t for t, _ in items], [f"short.{k}" for k in range(WB.SHORT_N + 1)])
        self.assertEqual(items[0][1], dict(WB.workload())[WB.SHORT_BASE])
        for (_, a), (_, b) in zip(items, items[1:]):
            self.assertEqual(b[:len(a)], a)
            self.assertEqual(len(b), len(a) + 2)
            self.assertLess(len(b[-1]["content"]), 2000)       # a short read: ~20 lines

    def test_system_tag_and_alternation(self):
        tagged = WB.workload("w2b")
        self.assertTrue(tagged[0][1][0]["content"].endswith("Session group: w2b."))
        self.assertEqual(WB.workload()[0][1][0]["content"], WB.atb.SYSTEM)     # agent_turns_bench's own, untagged
        saved = WB.atb.ROOT
        built = WB.build_turns(ROOT)
        self.assertIs(WB.atb.ROOT, saved)                                       # the module's root is put back
        self.assertEqual(WB.messages_hash(built), WB.messages_hash(WB.atb.turns()))
        self.assertEqual(WB.workload(None, built), built)
        order = WB.alternate({"A": [1, 2], "B": [10, 20, 30]})
        self.assertEqual(order, [("A", 1), ("B", 10), ("A", 2), ("B", 20), ("B", 30)])
        with self.assertRaises(ValueError):
            WB.parse_split("01/23")


class Matching(unittest.TestCase):
    def test_records_by_marker_with_continued_ids(self):
        rows = [{"marker": "wsb/1/P2/r0/w2b/A/000"}, {"marker": "wsb/1/P2/r0/w2b/B/001"}, {"marker": "wsb/none"}]
        recs = [{"id": "r1-2", "client": "wsb/1/P2/r0/w2b/A/000", "time": 2, "reused": 5},
                {"id": "r1", "client": "wsb/1/P2/r0/w2b/A/000", "time": 1, "reused": 100, "prompt_ms": 7.0},
                {"id": "r2", "client": "wsb/1/P2/r0/w2b/B/001", "time": 1, "reused": 0},
                {"id": "r3", "client": "Python-urllib/3.11", "time": 0}]
        WB.match_records(rows, recs)
        self.assertEqual(rows[0]["server"]["id"], "r1")
        self.assertEqual(rows[0]["server"]["reused"], 100)
        self.assertEqual(rows[0]["server_n"], 2)
        self.assertEqual(rows[0]["server_all"][0]["id"], "r1-2")
        self.assertEqual(rows[1]["server"]["id"], "r2")
        self.assertIsNone(rows[2]["server"])

    def test_engine_legs_two_users_interleaved(self):
        log = (
            "boot noise\n"
            "strata serve: prompt 30000 tokens = 2000 reused + 28000 of 28000 read in 9000 ms (3111.1 tok/s), 12 "
            "generated in 300 ms (40.0 tok/s), drafts accepted 8 of 10, 3 checkpoints (cancelled)\n"
            "strata serve: request metrics: decode entries 900 = primary 500 + secondary 200 + pcie 100 + cpu 100; "
            "cpu experts 40.5 ms; nvme loads 0; private commit 50.00 GiB\n"
            "strata batch: slot 1 takes 41001 tokens\n"
            "strata serve: prompt 41000 tokens = 40000 reused + 1000 read in 1500 ms (666.7 tok/s), 1 generated in "
            "0 ms (0.0 tok/s), drafts accepted 0 of 0, 1 checkpoints\n"
            "strata batch: slot 0 takes 30013 tokens\n"
            "strata serve: prompt 30012 tokens = 30012 reused + 0 read in 20 ms (0.0 tok/s), 1 generated in 0 ms "
            "(0.0 tok/s), drafts accepted 0 of 0, 1 checkpoints\n"
            "strata serve: prompt 30500 tokens = 30000 reused + 500 read in 700 ms (714.3 tok/s), 256 generated in "
            "4000 ms (64.0 tok/s), drafts accepted 100 of 150, 2 checkpoints\n"
            "strata serve: request metrics: decode entries 5000 = primary 3000 + secondary 1000 + pcie 500 + cpu 500; "
            "cpu experts 200.0 ms; nvme loads 0; private commit 50.00 GiB\n")
        legs = WB.parse_legs(log)
        self.assertEqual([lg["prompt"] for lg in legs], [30000, 41000, 30012, 30500])
        self.assertEqual(legs[0]["entries"], 900)
        self.assertTrue(legs[0]["cancelled"])
        self.assertEqual(legs[1]["slot"], 1)
        self.assertEqual(legs[2]["slot"], 0)
        self.assertNotIn("entries", legs[1])
        p3 = legs[3]["pos"]
        rows = [{"user": "A", "t_send": 1.0, "prompt_tokens": 30000, "completion_tokens": 256, "log_off0": 0},
                {"user": "B", "t_send": 1.5, "prompt_tokens": 41000, "completion_tokens": 256, "log_off0": 5},
                {"user": "A", "t_send": 9.0, "prompt_tokens": 30500, "completion_tokens": 256, "log_off0": p3}]
        WB.assign_legs(rows, legs)
        self.assertEqual([lg["prompt"] for lg in rows[0]["legs"]], [30000, 30012])
        self.assertEqual(rows[0]["engine_read_ms"], 9000.0)
        self.assertEqual(rows[0]["engine_tiers_solo_legs"]["cpu"], 100)
        self.assertEqual([lg["prompt"] for lg in rows[1]["legs"]], [41000])
        self.assertIsNone(rows[1]["engine_tiers_solo_legs"])
        self.assertEqual([lg["prompt"] for lg in rows[2]["legs"]], [30500])   # its own, not A's earlier 30012 leg
        self.assertFalse(any(r["legs_ambiguous"] for r in rows))
        self.assertAlmostEqual(WB.reused_fraction(rows[1])[0], 40000 / 41000)
        self.assertEqual(WB.reused_fraction(rows[1])[1], "engine")

    def test_a_read_that_gave_way_is_summed(self):
        # A's 60000-token read gives way (BYIELD) after 12288 tokens to B's short prompt, then goes on in slot 1
        log = (
            b"\xff\xfe not utf-8 \x80\n"
            b"strata serve: prompt 60000 tokens = 1000 reused + 12288 of 59000 read in 4000 ms (3072.0 tok/s), "
            b"0 generated in 0 ms (0.0 tok/s), drafts accepted 0 of 0, 1 checkpoints (cancelled)\n"
            b"strata batch: slot 0 takes 2001 tokens\n"
            b"strata serve: prompt 2000 tokens = 1000 reused + 1000 read in 500 ms (2000.0 tok/s), 1 generated in "
            b"0 ms (0.0 tok/s), drafts accepted 0 of 0, 1 checkpoints\n"
            b"strata batch: slot 1 takes 60001 tokens\n"
            b"strata serve: prompt 60000 tokens = 13288 reused + 46712 read in 15000 ms (3114.1 tok/s), 1 generated "
            b"in 0 ms (0.0 tok/s), drafts accepted 0 of 0, 2 checkpoints\n")
        legs = WB.parse_legs(log)
        self.assertEqual(legs[0]["pos"], log.index(b"strata serve: prompt 60000"))    # byte offsets, bad bytes too
        self.assertEqual(legs[2]["pos"], log.index(b"strata serve: prompt 60000 tokens = 13288"))
        rows = [{"user": "A", "t_send": 1.0, "prompt_tokens": 60000, "completion_tokens": 256, "log_off0": 0},
                {"user": "B", "t_send": 2.0, "prompt_tokens": 2000, "completion_tokens": 256, "log_off0": 0}]
        WB.assign_legs(rows, legs)
        a, b = rows
        self.assertEqual([lg["prompt"] for lg in a["legs"]], [60000, 60000])
        self.assertEqual(a["engine_read_ms"], 19000.0)
        self.assertEqual(a["engine_read"], 12288 + 46712)
        self.assertEqual(a["engine_read_legs"], 2)
        self.assertEqual(a["engine_reused"], 1000)                    # what the cache held when it arrived
        self.assertEqual([lg["prompt"] for lg in b["legs"]], [2000])
        self.assertFalse(a["legs_ambiguous"] or b["legs_ambiguous"])


class Stub:
    """A threaded /v1/chat/completions that streams SSE like serve/server.py (empty role chunk, keep-alive comments,
    reasoning, content, a last chunk with usage) and a /metrics keyed by User-Agent."""

    def __init__(self, read_s=0.3, tokens=5, tok_s=0.05):
        self.records, self.lock = [], threading.Lock()
        self.active, self.max_active = 0, 0
        stub = self

        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_GET(self):
                with stub.lock:
                    body = json.dumps({"requests": list(reversed(stub.records))}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                t0 = time.time()
                with stub.lock:
                    stub.active += 1
                    stub.max_active = max(stub.max_active, stub.active)
                    stub.records.append({"id": f"r{len(stub.records)}", "client": self.headers.get("User-Agent"),
                                         "time": t0, "prompt_ms": 1.0, "reused": 3,
                                         "body_slot": req.get("strata_cache_slot", "absent"),
                                         "body_tools": "tools" in req, "body_effort": req.get("reasoning_effort"),
                                         "body_max_tokens": req.get("max_tokens"),
                                         "body_kwargs": req.get("chat_template_kwargs")})
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()

                def send(obj):
                    self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
                    self.wfile.flush()

                def chunk(delta, finish=None):
                    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

                send(chunk({"role": "assistant", "content": ""}))
                time.sleep(stub_read / 2)
                self.wfile.write(b": ping\n\n")
                self.wfile.flush()
                time.sleep(stub_read / 2)
                send(chunk({"reasoning_content": "think"}))
                for _ in range(stub_tokens - 1):
                    time.sleep(stub_tok)
                    send(chunk({"content": "x"}))
                pt = len(json.dumps(req["messages"]))
                last = chunk({}, "length")
                last["usage"] = {"prompt_tokens": pt, "completion_tokens": stub_tokens, "total_tokens": pt + stub_tokens,
                                 "prompt_tokens_details": {"cached_tokens": 3}}
                last["timings"] = {"predicted_per_second": 9999}
                send(last)
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                with stub.lock:
                    stub.active -= 1
                self.close_connection = True

        stub_read, stub_tokens, stub_tok = read_s, tokens, tok_s
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def tiny_users():
    msgs = lambda u, i: [{"role": "system", "content": "s"}, {"role": "user", "content": f"{u}{i}" * (i + 1)}]  # noqa: E731
    return {"A": [(f"a.{i}", msgs("a", i)) for i in range(3)], "B": [(f"b.{i}", msgs("b", i)) for i in range(4)]}


class Schedulers(unittest.TestCase):
    def setUp(self):
        self.stub = Stub()

    def tearDown(self):
        self.stub.close()

    def mk(self, user, seq):
        return f"wsb/t/X/r0/m/{user}/{seq:03d}"

    def test_w2b_overlaps_both_users(self):
        rows = WB.run_concurrent(self.stub.port, tiny_users(), self.mk, max_new=8, seed=1)
        self.assertEqual(len(rows), 7)
        self.assertEqual(self.stub.max_active, 2)
        WB.mark_overlap(rows)
        self.assertTrue(any(r["overlap_s"] > 0.1 for r in rows if r["user"] == "A"))
        self.assertTrue(any(r["other_in_flight_at_send"] for r in rows))
        for u in "AB":                                   # each user waits for its own previous answer
            ur = [r for r in rows if r["user"] == u]
            for r1, r2 in zip(ur, ur[1:]):
                self.assertGreaterEqual(r2["t_send"], r1["t_end"])
        WB.match_records(rows, WB.fetch_metrics(self.stub.port))
        self.assertTrue(all(r["server"] and r["server"]["client"] == r["marker"] for r in rows))
        self.assertEqual(len({r["server"]["id"] for r in rows}), 7)

    def test_w2a_alternates_without_overlap(self):
        rows = WB.run_sequential(self.stub.port, WB.alternate(tiny_users()), self.mk, max_new=8, seed=1)
        self.assertEqual([r["user"] for r in rows], ["A", "B", "A", "B", "A", "B", "B"])
        self.assertEqual(self.stub.max_active, 1)
        WB.mark_overlap(rows)
        self.assertTrue(all(r["overlap_s"] == 0 for r in rows))

    def test_pin_sends_each_users_slot(self):
        rows = WB.run_sequential(self.stub.port, WB.send_order("w2a", tiny_users(), pin=True), self.mk, max_new=8,
                                 seed=1)
        by = {x["client"]: x for x in self.stub.records}
        for r in rows:
            self.assertEqual(by[r["marker"]]["body_slot"], {"A": 0, "B": 1}[r["user"]])
            self.assertEqual(r["cache_slot"], {"A": 0, "B": 1}[r["user"]])
        self.stub.records.clear()
        WB.run_sequential(self.stub.port, WB.send_order("w2a", tiny_users(), pin=False), self.mk, max_new=8, seed=1)
        self.assertTrue(all(x["body_slot"] == "absent" for x in self.stub.records))     # the server's rule decides
        self.stub.records.clear()
        WB.run_concurrent(self.stub.port, WB.pinned_users(tiny_users(), True), self.mk, max_new=8, seed=1)
        self.assertEqual({x["body_slot"] for x in self.stub.records}, {0, 1})

    def test_w2s_side_requests_between_turns(self):
        order = WB.send_order("w2s", tiny_users(), pin=True)
        self.assertEqual([u for u, _ in order][:6], ["A", "A~side", "B", "B~side", "A", "A~side"])
        self.assertEqual(len(order), 14)
        rows = WB.run_sequential(self.stub.port, order, self.mk, max_new=8, seed=1)
        by = {x["client"]: x for x in self.stub.records}
        for r in rows:
            got = by[r["marker"]]
            if r["user"].endswith("~side"):
                self.assertEqual(got["body_slot"], "absent")               # side requests are never pinned
                self.assertFalse(got["body_tools"])
                self.assertIsNone(got["body_effort"])
                self.assertEqual(got["body_max_tokens"], 32)
                self.assertEqual(got["body_kwargs"], {"enable_thinking": False})
                self.assertTrue(r["tag"].endswith(".side"))
            else:
                self.assertTrue(got["body_tools"])
                self.assertEqual(got["body_effort"], "high")

    def test_side_heads_differ_and_are_short(self):
        users = WB.split_users(WB.workload(), "01/234")
        sides = [WB.side_item(it) for it in users["A"] + users["B"]]
        heads = [s[1][0]["content"][:1500] for s in sides]                  # ~ the first 512 tokens' text
        self.assertEqual(len(set(heads)), len(heads))                        # a new prompt family every time
        for _, msgs, _extra in sides:
            n = len(msgs[0]["content"]) / 4                                   # characters / 4
            self.assertTrue(250 <= n <= 650, n)
            self.assertNotIn(WB.atb.SYSTEM[:200], msgs[0]["content"])        # not the agent's family

    def test_client_clock_not_timings(self):
        r = WB.post_stream(self.stub.port, tiny_users()["A"][0][1], max_new=8, seed=1, marker="m")
        self.assertIsNone(r["error"])
        self.assertEqual(r["first_kind"], "reasoning")      # the empty role chunk and the ping are not a token
        self.assertGreaterEqual(r["ttft_s"], 0.25)
        self.assertEqual(r["completion_tokens"], 5)
        self.assertEqual(r["cached_tokens"], 3)
        self.assertLess(r["client_decode_tok_s"], 100)      # ~4 tokens in ~0.2 s, not the stub's 9999 timings
        self.assertEqual(r["timings_untrusted"]["predicted_per_second"], 9999)
        self.assertEqual(r["reply_chars"], len("think") + 4)                 # reasoning + 4 content chunks of "x"
        self.assertEqual(len(r["reply_sha"]), 16)

    def test_summary_prints(self):
        rows = WB.run_concurrent(self.stub.port, tiny_users(), self.mk, max_new=8, seed=1)
        WB.mark_overlap(rows)
        WB.assign_legs(rows, [])
        out = []
        for arm in ("P2", "P1"):
            for r in rows:
                out.append({**r, "kind": "request", "arm": arm, "round": 0, "mode": "w2b", "server": None})
        text = WB.summarize(out)
        self.assertIn("w2b arm P2", text)
        self.assertIn("both", text)
        self.assertIn("ratio to P2", text)


DUMMY = """
import subprocess, sys, time
if sys.argv[1] == "parent":
    child = subprocess.Popen([sys.executable, __file__, "child"])
    print(child.pid, flush=True)
time.sleep(120)
"""


class TreeKill(unittest.TestCase):
    def test_kill_tree_ends_parent_and_grandchild(self):
        import psutil
        with tempfile.TemporaryDirectory() as d:
            script = Path(d) / "dummy.py"
            script.write_text(DUMMY, encoding="utf-8")
            flags = (WB.CREATE_NO_WINDOW | WB.CREATE_NEW_PROCESS_GROUP) if WB.WIN else 0
            t0 = time.time()
            p = subprocess.Popen([sys.executable, str(script), "parent"], stdout=subprocess.PIPE,
                                 creationflags=flags, text=True)
            child_pid = int(p.stdout.readline())
            child = psutil.Process(child_pid)
            self.assertTrue(child.is_running())
            res = WB.kill_tree(p.pid, timeout=20, since=t0)
            p.wait(10)
            p.stdout.close()
            self.assertEqual(res["leftover"], [])
            # the tree: the parent, its child and (Windows, CREATE_NO_WINDOW) the hidden console's conhost.exe
            self.assertLessEqual({p.pid, child_pid}, {x["pid"] for x in res["snapshot"]})
            self.assertLessEqual({x["name"].lower() for x in res["snapshot"]}, {"python.exe", "conhost.exe", "python"})
            self.assertFalse(child.is_running())
            self.assertEqual(res["late"], [])               # no strata.exe started by this test

    def test_late_kill_only_this_boots(self):
        """A process the snapshot missed is ended only when it is this boot's (its command line names the boot dir),
        never an unrelated one started after the boot.  python.exe stands in for strata.exe."""
        import psutil
        with tempfile.TemporaryDirectory() as d:
            boot = Path(d) / "boot-dir"
            boot.mkdir()
            flags = (WB.CREATE_NO_WINDOW | WB.CREATE_NEW_PROCESS_GROUP) if WB.WIN else 0
            sleeper = [sys.executable, "-c", "import time, sys; time.sleep(120)"]
            t0 = time.time()
            root = subprocess.Popen(sleeper, creationflags=flags)
            mine = subprocess.Popen(sleeper + [str(boot / "x.bin")], creationflags=flags)     # not in root's tree
            other = subprocess.Popen(sleeper + ["unrelated"], creationflags=flags)
            try:
                time.sleep(0.5)
                res = WB.kill_tree(root.pid, timeout=10, since=t0, scope_dir=boot, names=("python.exe", "python"))
                self.assertEqual([x["pid"] for x in res["late"]], [mine.pid])
                self.assertEqual(res["leftover"], [])
                mine.wait(10)
                self.assertTrue(psutil.Process(other.pid).is_running())
            finally:
                for p in (root, mine, other):
                    if p.poll() is None:
                        p.kill()
                        p.wait(10)

    def test_ram_wait_returns_at_once_when_reached(self):
        ok, avail, waited = WB.wait_ram(0, timeout=1)
        self.assertTrue(ok)
        self.assertGreater(avail, 0)
        self.assertLess(waited, 1)
        ok, _, waited = WB.wait_ram(1 << 62, timeout=0.3, poll=0.1)
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main(verbosity=2)
