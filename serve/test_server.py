"""serve/test_server.py - the max tokens budget over both APIs, against the mock engine (no GPU, no pack).

    python -m unittest serve.test_server -v
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import threading
import time
import queue
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import CTX_SLACK, ByteTokenizer, EngineDied, GpuBusy, MockEngine, Service, StrataEngine, request_timings, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CTX = 4096
# longer than the old 1024 fallback, one token per byte, and not a loop: "x" * 2000 is one, and the serving loop
# guard (serve/loop_guard.py, xeno) rightly stops it at 512 characters
ANSWER = "".join("abcdefghijklmnopqrstuvwxyz"[(i * 7919 + i * i * 104729) % 26] for i in range(2000))


class RecordingEngine(MockEngine):
    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.last_max_new = max_new
        self.last_sampling = sampling
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class MaxTokens(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = RecordingEngine(tok, "</think>\n\n" + ANSWER, max_context=CTX)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def call(self, api, text="hi", **budget):
        """-> (status, body, prompt tokens, completion tokens); `budget` is merged into the request as given."""
        msgs = [{"role": "user", "content": text}]
        if api == "openai":
            s, b = self.post("/v1/chat/completions", {"model": "m", "messages": msgs, **budget})
            u = b.get("usage", {})
            return s, b, u.get("prompt_tokens"), u.get("completion_tokens")
        s, b = self.post("/v1/messages", {"model": "m", "messages": msgs, **budget})
        u = b.get("usage", {})
        return s, b, u.get("input_tokens"), u.get("output_tokens")

    def test_cache_slot_validation_precedes_stream_headers(self):
        for api in ("openai", "anthropic"):
            for stream in (False, True):
                for slot in (-1, 4, True, "1", 1.5, None):
                    with self.subTest(api=api, stream=stream, slot=slot):
                        status, body, _, _ = self.call(api, max_tokens=8, stream=stream, strata_cache_slot=slot)
                        self.assertEqual(status, 400)
                        self.assertIn("strata_cache_slot", body["error"]["message"])

    def test_cache_slot_reaches_engine_over_both_apis(self):
        for api in ("openai", "anthropic"):
            status, _, _, _ = self.call(api, max_tokens=8, strata_cache_slot=3)
            self.assertEqual(status, 200)
            self.assertEqual(self.engine.last_sampling["strata_cache_slot"], 3)

    def test_layer_split_rejects_nonzero_slot_before_streaming(self):
        previous = getattr(self.engine, "info", None)
        self.engine.info = {"cache_slots": "1"}
        try:
            status, body, _, _ = self.call("openai", max_tokens=8, stream=True, strata_cache_slot=1)
            self.assertEqual(status, 400)
            self.assertIn("single-GPU", body["error"]["message"])
            self.assertEqual(self.call("openai", max_tokens=8, strata_cache_slot=0)[0], 200)
        finally:
            if previous is None:
                del self.engine.info
            else:
                self.engine.info = previous

    def test_unset_budget_is_the_rest_of_the_context(self):
        cases = {"openai": [{"max_tokens": -1}, {"max_tokens": 0}, {}, {"max_tokens": None},
                            {"max_completion_tokens": -1}, {"max_completion_tokens": None, "max_tokens": None}],
                 "anthropic": [{"max_tokens": -1}, {"max_tokens": 0}, {}, {"max_tokens": None}]}
        for api, budgets in cases.items():
            for budget in budgets:
                with self.subTest(api=api, budget=budget):
                    s, b, pt, ct = self.call(api, **budget)
                    self.assertEqual(s, 200, b)
                    self.assertEqual(self.engine.last_max_new, CTX - CTX_SLACK - pt)
                    self.assertGreater(ct, 1024)          # the whole answer, not cut at the old 1024 fallback

    def test_explicit_budget_is_honoured(self):
        for api, budget in [("openai", {"max_tokens": 50}), ("openai", {"max_completion_tokens": 50}),
                            ("openai", {"max_completion_tokens": 50, "max_tokens": 9}),
                            ("anthropic", {"max_tokens": 50}), ("openai", {"max_tokens": 1500}),
                            ("anthropic", {"max_tokens": 1500})]:
            with self.subTest(api=api, budget=budget):
                want = budget.get("max_completion_tokens") or budget["max_tokens"]
                s, b, _, ct = self.call(api, **budget)
                self.assertEqual(s, 200, b)
                self.assertEqual(self.engine.last_max_new, want)
                self.assertEqual(ct, want)

    def test_explicit_budget_over_the_context_is_rejected(self):
        for api in ("openai", "anthropic"):
            with self.subTest(api=api):
                s, b, _, _ = self.call(api, max_tokens=CTX)
                self.assertEqual(s, 400)
                self.assertIn("exceeds the context", b["error"]["message"])

    def test_unset_budget_with_a_near_full_prompt(self):
        _, _, pt0, _ = self.call("openai", max_tokens=1)
        overhead = pt0 - len("hi")                  # the template's tokens around the user text
        for api in ("openai", "anthropic"):
            _, _, pa, _ = self.call(api, max_tokens=1)
            over = pa - pt0                          # the Anthropic template may differ slightly
            with self.subTest(api=api, room=5):     # a few tokens left: the budget is exactly those
                text = "y" * (CTX - CTX_SLACK - overhead - over - 5)
                s, b, pt, ct = self.call(api, text=text, max_tokens=-1)
                self.assertEqual(s, 200, b)
                self.assertEqual(self.engine.last_max_new, 5)
                self.assertEqual(ct, 5)
            with self.subTest(api=api, room=0):     # nothing left: rejected, not truncated
                text = "y" * (CTX - CTX_SLACK - overhead - over)
                s, b, _, _ = self.call(api, text=text)
                self.assertEqual(s, 400, b)
                self.assertIn("no room to answer", b["error"]["message"])

    def test_debug_log_shows_the_resolved_budget(self):
        import contextlib
        import io
        os.environ["STRATA_DEBUG"] = "1"
        try:
            for api in ("openai", "anthropic"):
                with self.subTest(api=api):
                    out = io.StringIO()
                    with contextlib.redirect_stdout(out):
                        _, _, pt, _ = self.call(api, max_tokens=-1)
                    self.assertIn(f"max_new={CTX - CTX_SLACK - pt} ", out.getvalue())
        finally:
            del os.environ["STRATA_DEBUG"]


class FitMaxTokens(unittest.TestCase):
    """PR #24: --fit-max-tokens clamps an explicit budget that overshoots the context instead of a 400."""

    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = RecordingEngine(tok, "</think>\n\n" + ANSWER, max_context=CTX)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"), fit_max_tokens=True)
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    post = MaxTokens.post
    call = MaxTokens.call

    def test_overshoot_is_clamped_to_the_room(self):
        for api in ("openai", "anthropic"):
            with self.subTest(api=api):
                s, b, pt, ct = self.call(api, max_tokens=CTX)
                self.assertEqual(s, 200, b)
                self.assertEqual(self.engine.last_max_new, CTX - CTX_SLACK - pt)

    def test_a_budget_that_fits_is_unchanged(self):
        s, b, _, ct = self.call("openai", max_tokens=50)
        self.assertEqual(s, 200, b)
        self.assertEqual(self.engine.last_max_new, 50)

    def test_no_room_is_still_a_400(self):
        _, _, pt0, _ = self.call("openai", max_tokens=1)
        overhead = pt0 - len("hi")
        s, b, _, _ = self.call("openai", text="y" * (CTX - CTX_SLACK - overhead), max_tokens=100)
        self.assertEqual(s, 400, b)
        self.assertIn("no room to answer", b["error"]["message"])


class ImageMarkers(unittest.TestCase):
    """#150: the text "<|image_pad|>" inside a message is text, not an image's place."""

    class FakeVision:
        def __init__(self, d):
            self.dir = Path(d)
            self.rows = self.dir / "img.sve"
            self.rows.write_bytes(b"rows")

        def encode(self, source):
            return self.rows, 3

    def test_literal_marker_with_an_image(self):
        import tempfile
        tok = ByteTokenizer()
        with tempfile.TemporaryDirectory() as d:
            svc = Service(MockEngine(tok, "ok", max_context=CTX), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"),
                          vision=self.FakeVision(d))
            pad = tok.encode("<|image_pad|>", parse_special=True)[0]
            for text in ("the docs say <|image_pad|> marks an image", "plain"):
                with self.subTest(text=text):
                    msgs = [{"role": "user", "content": [{"type": "text", "text": text},
                                                         {"type": "image", "source": "x.png"}]}]
                    ids, _, _ = svc.prepare(msgs, None, {})
                    self.assertEqual(ids.count(pad), 3)          # the image's three rows, nothing else
                    self.assertIn("<|image_pad|> marks" if "docs" in text else "plain", tok.decode(ids))
            svc.embeddings.path.unlink(missing_ok=True)


class StatusNeedsTheKey(unittest.TestCase):
    """#212: /status shows the end of the answer being written, so it needs the key like /v1/*."""

    def test_status(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok", max_context=CTX), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.api_key = "k3y"
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}/status"
        try:
            with self.assertRaises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(base, timeout=10)
            self.assertEqual(e.exception.code, 401)
            e.exception.close()
            req = urllib.request.Request(base, headers={"Authorization": "Bearer k3y"})
            with urllib.request.urlopen(req, timeout=10) as r:
                self.assertEqual(r.status, 200)
                self.assertNotIn("tail", json.loads(r.read()))
        finally:
            httpd.shutdown()
            httpd.server_close()


class ToolCallTerminators(unittest.TestCase):
    """#210: a value that contains </parameter> or </tool_call> (a file documenting the call format) is kept whole."""
    CONTENT = ("Close each value with </parameter> and the call with </function></tool_call>.\n"
               "<parameter=x>\nnot a parameter\n</parameter>\nend")
    SCHEMA = [{"name": "write", "parameters": {"properties": {"path": {"type": "string"},
                                                              "content": {"type": "string"}}}}]

    def run_parser(self, stream_tools, step):
        from serve.frontend import OutputParser
        text = ("</think>\n\n<tool_call>\n<function=write>\n<parameter=path>\ndoc.md\n</parameter>\n"
                f"<parameter=content>\n{self.CONTENT}\n</parameter>\n</function>\n</tool_call>")
        p = OutputParser(thinking=True, tools=self.SCHEMA, stream_tools=stream_tools)
        evs = []
        for i in range(0, len(text), step):
            evs += p.feed(text[i:i + step])
        evs += p.finish()
        return evs

    def test_values_keep_the_terminators(self):
        for stream_tools in (False, True):
            for step in (1, 7, 10_000):
                with self.subTest(stream_tools=stream_tools, step=step):
                    evs = self.run_parser(stream_tools, step)
                    calls = [e.call for e in evs if e.kind == "tool_call"]
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(calls[0].arguments, {"path": "doc.md", "content": self.CONTENT})
                    self.assertFalse([e for e in evs if e.kind == "content" and e.text.strip()])
                    if stream_tools:
                        streamed = "".join(e.text for e in evs if e.kind == "tool_args")
                        self.assertEqual(json.loads(streamed), {"path": "doc.md", "content": self.CONTENT})


class ClientShapes(unittest.TestCase):
    """What real clients send: Claude Code posts /v1/messages?beta=true (issue #55) and puts hook context into the
    conversation as a mid-conversation system message (issue #56); some OpenAI clients send a late developer message."""

    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = RecordingPrompt(tok, "</think>\n\n2", max_context=CTX)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def prompt_text(self):
        return bytes(i for i in self.engine.last_ids if i < 256).decode("utf-8", "replace")

    def test_query_string(self):
        body = {"model": "x", "max_tokens": 20, "messages": [{"role": "user", "content": "hi"}]}
        for path in ("/v1/messages?beta=true", "/v1/chat/completions?api-version=1", "/v1/messages/?beta=true"):
            status, b = self.post(path, body)
            self.assertEqual(status, 200, (path, b))
        status, _ = self.post("/v1/nothing?beta=true", body)
        self.assertEqual(status, 404)

    def test_anthropic_mid_conversation_system(self):
        status, b = self.post("/v1/messages?beta=true", {
            "model": "x", "max_tokens": 50,
            "system": [{"type": "text", "text": "You are terse."}],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "1+1? digits only"}]},
                {"role": "system", "content": [{"type": "text", "text": "<system-reminder>answer in digits</system-reminder>"}]}]})
        self.assertEqual(status, 200, b)
        text = self.prompt_text()
        self.assertIn("You are terse.", text)
        self.assertIn("<system-reminder>answer in digits</system-reminder>", text)
        self.assertLess(text.index("You are terse."), text.index("1+1?"))          # the first system stays first
        self.assertLess(text.index("1+1?"), text.index("answer in digits"))        # the late one stays in place

    def test_openai_late_developer_and_system(self):
        status, b = self.post("/v1/chat/completions", {
            "model": "x", "max_tokens": 50,
            "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hello"},
                         {"role": "assistant", "content": "hi"}, {"role": "developer", "content": "Now use digits."},
                         {"role": "system", "content": "Also this."}, {"role": "user", "content": "1+1?"}]})
        self.assertEqual(status, 200, b)
        text = self.prompt_text()
        for part in ("Be brief.", "Now use digits.", "Also this.", "1+1?"):
            self.assertIn(part, text)

    def test_leading_system_unchanged(self):
        from serve.frontend import anthropic_to_messages, openai_to_messages
        msgs, _, _ = openai_to_messages({"messages": [{"role": "developer", "content": "D"}, {"role": "user", "content": "u"}]})
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])
        msgs, _, _ = anthropic_to_messages({"system": "S", "messages": [{"role": "user", "content": "u"}]})
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])


class SamplingKeys(unittest.TestCase):
    """The GEN line's sampling keys: top_k 0 ("off") or wider than the engine's 64 get the widest list, 64 (they used
    to fall back to the engine default 20); a penalty always carries its window."""

    def keys(self, **sampling):
        return StrataEngine.sampling_keys(sampling).split()

    def test_top_k(self):
        self.assertIn("top_k=10", self.keys(temperature=0.7, top_k=10))
        self.assertIn("top_k=64", self.keys(temperature=0.7, top_k=64))
        self.assertIn("top_k=64", self.keys(temperature=0.7, top_k=0))
        self.assertIn("top_k=64", self.keys(temperature=0.7, top_k=100))
        for bad in (-1, True, 2.5, "20"):
            self.assertFalse([k for k in self.keys(temperature=0.7, top_k=bad) if k.startswith("top_k=")], bad)

    def test_tune_keys(self):
        k = self.keys(temperature=0, strata_tune={"pcie_frac": 0.2, "spec_min_p": 0.7})
        self.assertIn("pcie_frac=0.2", k)
        self.assertIn("spec_min_p=0.7", k)
        bad = self.keys(strata_tune={"pcie_frac": 3, "spec_min_p": True, "pool_workers": 2})
        self.assertFalse([x for x in bad if x.split("=")[0] in ("pcie_frac", "spec_min_p", "pool_workers")])

    def test_penalty_window(self):
        self.assertIn("penalty_last_n=64", self.keys(presence_penalty=1.5))
        self.assertIn("penalty_last_n=4096", self.keys(repetition_penalty=1.1, penalty_last_n=4096))
        self.assertFalse([k for k in self.keys(temperature=0.7) if k.startswith("penalty")])


class GpuChoice(unittest.TestCase):
    """Issue #51: the config's \"gpu\" reaches the engine as CUDA_VISIBLE_DEVICES, numbered like nvidia-smi."""

    def test_env(self):
        from serve.server import child_env
        env = child_env({"gpu": 1})
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(env["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        plain = child_env({})                     # no choice: the environment as it was (existing installs)
        self.assertEqual(plain.get("CUDA_VISIBLE_DEVICES"), os.environ.get("CUDA_VISIBLE_DEVICES"))
        self.assertEqual(plain.get("CUDA_DEVICE_ORDER"), os.environ.get("CUDA_DEVICE_ORDER"))


class RecordingPrompt(MockEngine):
    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.last_ids = list(ids)
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class DyingEngine(MockEngine):
    """Issue #27: an engine that dies after a few tokens of its first answer, and comes back when restarted."""

    def __init__(self, tok, script, max_context):
        super().__init__(tok, script, max_context=max_context)
        self.dead, self.restarts, self.die_after = False, 0, 5

    def alive(self):
        return not self.dead

    def restart(self):
        self.dead, self.die_after = False, None
        self.restarts += 1

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        for i, t in enumerate(super().generate(ids, max_new, sampling, cancel, embeddings)):
            if self.die_after is not None and i == self.die_after:
                self.dead = True
                raise EngineDied("the engine stopped unexpectedly (exit code -9)")
            yield t


class EngineDeath(unittest.TestCase):
    """Issue #27: a dead engine is an error (not "length"), and the next request starts it again."""

    def test_error_then_restart(self):
        tok = ByteTokenizer()
        eng = DyingEngine(tok, "</think>\n\n" + ANSWER, max_context=CTX)
        svc = Service(eng, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            def post(body):
                req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
                try:
                    with urllib.request.urlopen(req, timeout=30) as r:
                        return r.status, r.read().decode()
                except urllib.error.HTTPError as e:
                    with e:
                        return e.code, e.read().decode()
            msgs = [{"role": "user", "content": "hi"}]
            code, text = post({"model": "m", "messages": msgs, "max_tokens": 50, "stream": True})
            self.assertEqual(code, 200)
            self.assertIn('"error"', text)
            self.assertIn("stopped unexpectedly", text)
            self.assertTrue(text.rstrip().endswith("data: [DONE]"))
            self.assertEqual(svc.metrics()["requests"][0]["finish"], "error")
            code, text = post({"model": "m", "messages": msgs, "max_tokens": 50})
            self.assertEqual(code, 200, text)
            self.assertEqual(eng.restarts, 1)
            self.assertEqual(json.loads(text)["usage"]["completion_tokens"], 50)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_engine_err_mid_stream(self):
        """The engine's ERR line after the stream started reaches the client as an error event (it used to be a
        400 written into the open stream, which clients read as an empty answer)."""
        class ErrEngine(MockEngine):
            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                yield None                                  # a prompt-progress heartbeat: the stream has started
                raise ValueError("verify: layer 31 never rang (an illegal memory access was encountered)")

        tok = ByteTokenizer()
        svc = Service(ErrEngine(tok, ANSWER, max_context=CTX), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            for path, body in [("/v1/chat/completions", {"model": "m", "stream": True, "max_tokens": 20,
                                                          "messages": [{"role": "user", "content": "hi"}]}),
                               ("/v1/messages", {"model": "m", "stream": True, "max_tokens": 20,
                                                 "messages": [{"role": "user", "content": "hi"}]})]:
                req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    text = r.read().decode()
                self.assertIn("illegal memory access", text, path)
                self.assertNotIn("HTTP/1", text, path)
                self.assertEqual(svc.metrics()["requests"][0]["finish"], "error")
        finally:
            httpd.shutdown()
            httpd.server_close()


class LiveRate(unittest.TestCase):
    """The Monitor's Speed readout: live.tok_s is a rate, and a request that never got a DONE keeps no counters.

    It used to be `generated / (now - first_token)` - the mean since the first token, whose first sample is
    1/elapsed.  Against a paced engine that reads five-digit numbers for the first instant of every answer and
    undershoots for the first second after that.  It is now the rate over the last RATE_WINDOW_S, with the mean
    still available as `live.tok_s_mean` for anyone who wants it."""

    PACE_S = 0.02                    # 50 tokens/s: a 30-token answer takes about 0.6 s
    TOKENS = 30

    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = MockEngine(self.tok, "x" * self.TOKENS, max_context=CTX, delay_s=self.PACE_S)
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def metrics(self):
        with urllib.request.urlopen(self.base + "/metrics", timeout=10) as r:
            return json.loads(r.read())

    def test_prefill_rate_excludes_cached_tokens(self):
        import io
        import queue
        from types import SimpleNamespace
        engine = StrataEngine.__new__(StrataEngine)
        engine.proc = SimpleNamespace(stdin=io.StringIO(), poll=lambda: None)   # alive() asks it (#208)
        engine.lines = queue.Queue()
        engine.can_stop = False
        engine.max_context = 262144
        engine.prefill_tok_s_mean = 9999.0
        engine.lines.put("PP 10000 12000 2000 1000.0")  # 8000 cached, 2000 newly read in two seconds
        engine.lines.put("DONE 1 12000 4000 10 stop 0 0 8000")
        gen = engine.generate([1], 1, {}, threading.Event())
        self.assertIsNone(next(gen))
        self.assertEqual(engine.progress, (10000, 12000))
        self.assertEqual(engine.prefill_tok_s_mean, 1000.0)
        self.svc.engine = engine
        self.svc.status.update(busy=True, first_token=None)
        self.assertEqual(self.metrics()["live"]["prefill_tok_s_mean"], 1000.0)
        self.assertNotIn("prefill_tok_s", self.metrics()["live"])
        self.assertEqual(self.svc._prefill_tok_s_mean(), 1000.0)
        self.svc.status.update(first_token=time.time(), generated=1)
        self.assertEqual(self.svc._prefill_tok_s_mean(), 0.0)
        self.assertEqual(list(gen), [])
        timings = request_timings(12000, 1, engine.last)
        self.assertEqual(timings["prompt_per_second"], 1000.0)
        engine.lines.put("PP 8000 12000")
        engine.lines.put("DONE 0 12000 0 0 stop 0 0 12000")
        gen = engine.generate([1], 1, {}, threading.Event())
        next(gen)
        self.assertIsNone(engine.prefill_tok_s_mean)
        list(gen)
        self.svc.status["busy"] = False
        self.assertIsNone(self.metrics()["live"]["prefill_tok_s_mean"])

    def test_the_live_number_is_a_rate(self):
        live_samples, stop = [], threading.Event()

        def poll():                                   # what the Monitor polls, at 10 ms
            while not stop.is_set():
                live = self.metrics()["live"]
                if live["state"] == "generating" and live["tok_s"] is not None:
                    live_samples.append((live["generated"], live["tok_s"], live["tok_s_mean"]))
                time.sleep(0.01)

        body = json.dumps({"model": "m", "max_tokens": self.TOKENS, "temperature": 0,
                           "messages": [{"role": "user", "content": "hi"}]}).encode()
        watcher = threading.Thread(target=poll, daemon=True)
        watcher.start()
        t0 = time.time()
        try:
            with urllib.request.urlopen(urllib.request.Request(self.base + "/v1/chat/completions", data=body,
                                                               headers={"Content-Type": "application/json"}),
                                        timeout=30) as r:
                usage = json.loads(r.read())["usage"]
        finally:
            stop.set()
            watcher.join(2)
        true_rate = usage["completion_tokens"] / (time.time() - t0)
        self.assertGreaterEqual(len(live_samples), 3, "too few live readings to judge the readout")
        self.assertLess(max(s for _g, s, _m in live_samples), 4 * true_rate,
                        f"live.tok_s peaked at {max(s for _g, s, _m in live_samples):.1f} tok/s "
                        f"for a {true_rate:.1f} tok/s engine")
        self.assertEqual(self.metrics()["live"]["state"], "idle")
        self.assertIsNone(self.metrics()["live"]["tok_s"])

    def test_a_request_without_a_done_keeps_no_engine_counters(self):
        """An engine that dies mid-answer: the previous request's `last` must not become this row's decode rate."""
        class HalfDead(MockEngine):
            last = {"generated": 99, "prompt_tokens": 9, "prompt_ms": 10.0, "decode_ms": 100.0, "finish": "stop"}

            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                for i, t in enumerate(super().generate(ids, max_new, sampling, cancel, embeddings)):
                    if i == 3:
                        raise EngineDied("the engine stopped unexpectedly (exit code -9)")
                    yield t

        tok = ByteTokenizer()
        svc = Service(HalfDead(tok, "x" * self.TOKENS, max_context=CTX), tok,
                      ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            body = json.dumps({"model": "m", "max_tokens": self.TOKENS, "stream": True,
                               "messages": [{"role": "user", "content": "hi"}]}).encode()
            with urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions", data=body,
                                                               headers={"Content-Type": "application/json"}),
                                        timeout=30) as r:
                text = r.read().decode()
            self.assertIn('"error"', text)
            row = svc.metrics()["requests"][0]
            self.assertEqual(row["finish"], "error")
            self.assertEqual(row["output_tokens"], 3)
            self.assertIsNone(row["decode_tok_s"], "the previous request's counters were recorded as this one's")
            self.assertIsNone(row["engine_generated"])
        finally:
            httpd.shutdown()
            httpd.server_close()


class SharedSettings(unittest.TestCase):
    """The web app's "Use for other apps too": POST /settings makes its Chat settings every client's defaults."""

    @classmethod
    def setUpClass(cls):
        import tempfile

        class Sampled(RecordingEngine):
            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                self.last_sampling = dict(sampling or {})
                yield from super().generate(ids, max_new, sampling, cancel, embeddings)

        tok = ByteTokenizer()
        cls.engine = Sampled(tok, "</think>\n\nhello", max_context=CTX)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.tmp = tempfile.TemporaryDirectory()
        cls.svc.shared_path = os.path.join(cls.tmp.name, "strata-x.shared-settings.json")
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.tmp.cleanup()

    def req(self, path, body, headers=None, raw=None):
        h = {"Content-Type": "application/json", **(headers or {})}
        r = urllib.request.Request(self.base + path, data=raw if raw is not None else json.dumps(body).encode(), headers=h)
        try:
            with urllib.request.urlopen(r, timeout=30) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def chat(self, **extra):
        return self.req("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}], **extra})

    def tearDown(self):
        self.svc.set_shared(None)

    def test_other_apps_get_the_chat_settings(self):
        d = {"temperature": 0.3, "top_p": 0.9, "top_k": 10, "seed": 7, "max_tokens": 77,
             "reasoning_effort": "low", "experimental_speed_projection": False}
        code, b = self.req("/settings", {"defaults": d})
        self.assertEqual(code, 200, b)
        self.assertTrue(b["shared"])
        self.assertTrue(os.path.exists(self.svc.shared_path))
        code, _ = self.chat()                                        # a client that sets nothing
        self.assertEqual(code, 200)
        got = self.engine.last_sampling
        for k in ("temperature", "top_p", "top_k", "seed", "experimental_speed_projection"):
            self.assertEqual(got[k], d[k], k)
        self.assertEqual(self.engine.last_max_new, 77)
        code, _ = self.chat(temperature=0.9, max_tokens=5)          # its own values win
        self.assertEqual(self.engine.last_sampling["temperature"], 0.9)
        self.assertEqual(self.engine.last_max_new, 5)
        r = self.svc.with_shared({"messages": []}, "openai")
        self.assertEqual(r["reasoning_effort"], "low")
        self.assertEqual(self.svc.with_shared({"reasoning_effort": "high"}, "openai")["reasoning_effort"], "high")
        self.assertEqual(self.svc.with_shared({}, "anthropic")["output_config"], {"effort": "low"})

    def test_off_again(self):
        self.req("/settings", {"defaults": {"temperature": 0.3}})
        code, b = self.req("/settings", {"defaults": None})
        self.assertEqual((code, b["shared"]), (200, False))
        self.assertFalse(os.path.exists(self.svc.shared_path))
        self.chat()
        self.assertNotIn("temperature", self.engine.last_sampling)

    def test_only_strata_s_own_page_may_set_them(self):
        code, _ = self.req("/settings", None, {"Content-Type": "text/plain"}, raw=b'{"defaults": {"temperature": 1}}')
        self.assertEqual(code, 415)
        code, _ = self.req("/settings", {"defaults": {"temperature": 1}}, {"Origin": "http://evil.example"})
        self.assertEqual(code, 403)
        code, b = self.req("/settings", {"defaults": {"temperature": 9}})
        self.assertEqual(code, 400)
        self.assertIn("temperature", b["error"]["message"])
        self.assertEqual(self.svc.shared, {})
        host = self.base.split("://", 1)[1]
        code, _ = self.req("/settings", {"defaults": {"temperature": 1}}, {"Origin": "http://" + host})
        self.assertEqual(code, 200)

    def test_they_need_the_key_when_one_is_set(self):
        self.svc.api_key = "secret"
        try:
            self.assertEqual(self.req("/settings", {"defaults": {"temperature": 1}})[0], 401)
            self.assertEqual(self.req("/settings", {"defaults": {"temperature": 1}},
                                      {"Authorization": "Bearer secret"})[0], 200)
        finally:
            self.svc.api_key = ""


class WebApp(unittest.TestCase):
    """The web app (PR #22's dashboard idea, rebuilt): its page and files, and GET /metrics."""

    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.svc = Service(RecordingEngine(tok, "</think>\n\nhello", max_context=CTX), tok,
                          ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def get(self, path, headers=None):
        req = urllib.request.Request(self.base + path, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.headers.get("Content-Type", ""), r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers.get("Content-Type", ""), e.read()

    def test_page_and_files(self):
        code, ctype, body = self.get("/")
        self.assertEqual(code, 200)
        self.assertIn("text/html", ctype)
        self.assertIn(b"\"web/app.js\"", body)   # relative since #82 (works behind a path-prefixed proxy)
        for path, want in (("/web/app.js", "javascript"), ("/web/app.css", "text/css"), ("/web/tokens.css", "text/css"),
                           ("/web/components.css", "text/css"), ("/web/sprite.svg", "image/svg+xml")):
            with self.subTest(path=path):
                code, ctype, _ = self.get(path)
                self.assertEqual(code, 200)
                self.assertIn(want, ctype)

    def test_only_the_app_files_are_served(self):
        for path in ("/web/..%2Fserver.py", "/web/index.html", "/web/test.py", "/fonts/..%2F..%2Fsetup.py",
                     "/fonts/missing.woff2", "/fonts/x.ttf"):
            with self.subTest(path=path):
                self.assertEqual(self.get(path)[0], 404)

    def test_metrics(self):
        data = json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5}).encode()
        urllib.request.urlopen(urllib.request.Request(self.base + "/v1/chat/completions", data=data,
                                                      headers={"Content-Type": "application/json"}), timeout=10).read()
        code, ctype, body = self.get("/metrics")
        self.assertEqual(code, 200)
        m = json.loads(body)
        for key in ("engine", "live", "requests", "hardware", "hardware_static", "history"):
            self.assertIn(key, m)
        self.assertEqual(m["engine"]["max_context"], CTX)
        self.assertEqual(m["live"]["state"], "idle")
        self.assertEqual(m["requests"][0]["output_tokens"], 5)

    def test_model_discovery_and_props(self):
        svc = self.svc
        previous = svc.engine.max_context, svc.vision, svc.sampling_defaults, svc.shared
        try:
            svc.engine.max_context = 262144
            svc.sampling_defaults = {"temperature": 1.0, "repetition_penalty": 1.1}
            svc.shared = {"temperature": 0.7, "max_tokens": 4096}
            for vision in (None, object()):
                svc.vision = vision
                for path in ("/models", "/v1/models"):
                    code, _, body = self.get(path)
                    self.assertEqual(code, 200)
                    models = json.loads(body)["data"]
                    self.assertEqual(len(models), 1)
                    model = models[0]
                    self.assertEqual(model["id"], svc.model)
                    self.assertEqual(model["status"]["value"], "loaded")
                    self.assertEqual(model["meta"]["n_ctx"], 262144)
                    self.assertEqual(model["architecture"]["input_modalities"],
                                     ["text", "image"] if vision else ["text"])
                code, _, body = self.get("/props?model=" + svc.model + "&autoload=false")
                self.assertEqual(code, 200)
                props = json.loads(body)
                self.assertEqual(props["default_generation_settings"]["n_ctx"], 262144)
                self.assertEqual(props["default_generation_settings"]["params"],
                                 {"temperature": 0.7, "repeat_penalty": 1.1, "n_predict": 4096})
                self.assertEqual(props["chat_template"], (ROOT / "serve/chat_template.jinja").read_text(encoding="utf-8"))
                self.assertEqual(props["modalities"]["vision"], vision is not None)
                self.assertEqual(props["total_slots"], 1)
                self.assertFalse(props["models_autoload"])
            svc.shared = {}
            props = json.loads(self.get("/props")[2])
            self.assertEqual(props["default_generation_settings"]["params"]["n_predict"], -1)
            self.assertEqual(self.get("/props?model=not-loaded&autoload=true")[0], 404)
        finally:
            svc.engine.max_context, svc.vision, svc.sampling_defaults, svc.shared = previous

    def test_discovery_needs_the_api_key(self):
        self.svc.api_key = "secret"
        try:
            for path in ("/models", "/v1/models", "/props", "/slots"):
                self.assertEqual(self.get(path)[0], 401)
                self.assertEqual(self.get(path, {"Authorization": "Bearer secret"})[0], 200)
        finally:
            self.svc.api_key = ""

    def test_build_model_path_and_slot_status(self):
        engine = self.svc.engine
        engine.model_path = "models/example.gguf"
        engine.info = {"version": "0.1.21"}
        try:
            props = json.loads(self.get("/props")[2])
            self.assertEqual(props["model_path"], engine.model_path)
            self.assertEqual(props["build_info"], "Strata 0.1.21")
            for busy in (True, False):
                with self.svc.status_lock:
                    self.svc.status["busy"] = busy
                code, _, body = self.get("/slots")
                self.assertEqual(code, 200)
                self.assertEqual(json.loads(body), [{"id": 0, "n_ctx": CTX, "is_processing": busy}])
        finally:
            with self.svc.status_lock:
                self.svc.status["busy"] = False
            del engine.model_path, engine.info
        props = json.loads(self.get("/props")[2])
        self.assertNotIn("build_info", props)
        self.assertNotIn("model_path", props)

    def test_discovery_does_not_restart_a_dead_engine(self):
        self.svc.engine.alive = lambda: False
        try:
            for path in ("/models", "/v1/models"):
                code, _, body = self.get(path)
                self.assertEqual(code, 200)
                self.assertEqual(json.loads(body)["data"], [])
            self.assertEqual(self.get("/props")[0], 503)
            self.assertEqual(json.loads(self.get("/slots")[2]), [])
        finally:
            del self.svc.engine.alive

    def test_metrics_need_the_key_when_one_is_set(self):
        self.svc.api_key = "secret"
        try:
            self.assertEqual(self.get("/metrics")[0], 401)
            self.assertEqual(self.get("/metrics", {"Authorization": "Bearer secret"})[0], 200)
            self.assertEqual(self.get("/")[0], 200)                  # the page itself asks for the key
        finally:
            self.svc.api_key = ""


class ClockedEngine(MockEngine):
    """The mock engine with StrataEngine's clock: `last` as the engine's DONE line gives it, the conversation cache
    holding the first REUSED tokens of every prompt."""
    REUSED = 5

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        n = 0
        try:
            for t in super().generate(ids, max_new, sampling, cancel, embeddings):
                n += 1
                yield t
        finally:          # as StrataEngine reads its DONE line: also when the server closes the request at a stop token
            self.last = {"generated": n, "prompt_tokens": len(ids), "prompt_ms": 40.0, "decode_ms": 20.0 * n,
                         "finish": "stop", "reused": min(self.REUSED, len(ids)), "hits": 9, "lookups": 10}


class UsageAndStatus(unittest.TestCase):
    """What clients read besides the text: the part of the prompt the conversation cache held (OpenAI's
    prompt_tokens_details.cached_tokens, Anthropic's cache_read_input_tokens), llama.cpp's timings, GET /v1/status."""

    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = ClockedEngine(tok, "</think>\n\nok", max_context=CTX)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def request(self, path, body=None):
        req = urllib.request.Request(self.base + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read()

    def chat(self, path, stream=False):
        body = {"model": "x", "max_tokens": 20, "messages": [{"role": "user", "content": "hi"}], "stream": stream}
        status, raw = self.request(path, body)
        self.assertEqual(status, 200)
        if not stream:
            return json.loads(raw)
        return [json.loads(line[6:]) for line in raw.decode().splitlines()
                if line.startswith("data: {")]

    def test_openai(self):
        b = self.chat("/v1/chat/completions")
        u, t = b["usage"], b["timings"]
        self.assertEqual(u["prompt_tokens_details"]["cached_tokens"], ClockedEngine.REUSED)
        self.assertEqual(t["cache_n"], ClockedEngine.REUSED)
        self.assertEqual(t["prompt_n"] + t["cache_n"], u["prompt_tokens"])
        self.assertEqual(t["predicted_n"], u["completion_tokens"])
        self.assertAlmostEqual(t["prompt_per_second"], t["prompt_n"] / 0.040, delta=0.1)
        self.assertAlmostEqual(t["predicted_per_second"], 50.0, delta=0.1)            # 20 ms a token

    def test_openai_stream(self):
        last = self.chat("/v1/chat/completions", stream=True)[-1]
        self.assertEqual(last["usage"]["prompt_tokens_details"]["cached_tokens"], ClockedEngine.REUSED)
        self.assertEqual(last["timings"]["cache_n"], ClockedEngine.REUSED)

    def test_anthropic(self):
        u = self.chat("/v1/messages")["usage"]
        self.assertEqual(u["cache_read_input_tokens"], ClockedEngine.REUSED)
        self.assertEqual(u["input_tokens"] + u["cache_read_input_tokens"], len(self.engine.last_prompt))
        self.assertGreater(u["output_tokens"], 0)

    def test_v1_status(self):
        self.chat("/v1/chat/completions")
        status, raw = self.request("/v1/status")
        self.assertEqual(status, 200)
        s = json.loads(raw)
        self.assertEqual(s["model"], self.svc.model)
        self.assertEqual(s["context"]["max_positions"], CTX)
        self.assertEqual(s["concurrency"]["serving"], 1)
        self.assertFalse(s["vision"]["available"])
        self.assertEqual(s["activity"]["in_flight"], 0)
        self.assertGreaterEqual(s["activity"]["requests"], 1)
        self.assertEqual(s["last_timings"]["cache_n"], ClockedEngine.REUSED)
        self.assertIn("at", s["last_timings"])

    def test_no_clock(self):
        """An engine without a clock (MockEngine): no timings, nothing cached."""
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "</think>\n\nok", max_context=CTX), tok,
                      ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        try:
            data = json.dumps({"model": "x", "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions", data=data,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                b = json.loads(r.read())
            self.assertNotIn("timings", b)
            self.assertEqual(b["usage"]["prompt_tokens_details"]["cached_tokens"], 0)
            self.assertIsNone(svc.v1_status()["last_timings"])
        finally:
            httpd.shutdown()
            httpd.server_close()


class TimingsDrafts(unittest.TestCase):
    """`timings` carries the speculative draft counts (PR #83's fields) only when the engine reported them."""

    def test_draft_fields(self):
        base = {"prompt_ms": 100.0, "decode_ms": 200.0, "generated": 20, "reused": 4}
        t = request_timings(24, 20, dict(base, drafts_offered=15, drafts_accepted=11))
        self.assertEqual((t["draft_n"], t["draft_n_accepted"]), (15, 11))
        self.assertEqual((t["prompt_n"], t["cache_n"]), (20, 4))
        self.assertNotIn("draft_n", request_timings(24, 20, base))
        self.assertIsNone(request_timings(24, 20, {}))


class ToolResultContent(unittest.TestCase):
    """#46 (xeno): Claude Code's Read returns a PDF as a `document` block and a screenshot as an `image` block
    INSIDE a tool_result. The translation kept only the text parts, so the model never saw either."""

    REQ = {"messages": [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": [
        {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "PDF TEXT HERE"}},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
        {"type": "text", "text": "tool text"}]}]}]}

    def tool_message(self, vision):
        from serve.frontend import anthropic_to_messages
        msgs, _, _ = anthropic_to_messages(json.loads(json.dumps(self.REQ)), vision=vision)
        tools = [m for m in msgs if m["role"] == "tool"]
        self.assertEqual(len(tools), 1, msgs)
        return msgs, tools[0]

    @staticmethod
    def text(content):
        return content if isinstance(content, str) else "".join(i.get("text", "") for i in content)

    def test_a_pdf_returned_by_read_reaches_the_tool_message(self):
        for vision in (False, True):
            _, tool = self.tool_message(vision)
            self.assertIn("PDF TEXT HERE", self.text(tool["content"]))
            self.assertIn("tool text", self.text(tool["content"]))

    def test_a_screenshot_returned_by_read_reaches_the_prompt_with_vision(self):
        from serve.frontend import images_of
        msgs, tool = self.tool_message(vision=True)
        self.assertEqual(images_of(msgs), ["data:image/png;base64,AAAA"])
        prompt = ChatTemplate(ROOT / "serve" / "chat_template.jinja").render(msgs)
        response = prompt[prompt.index("<tool_response>"):prompt.index("</tool_response>")]
        self.assertIn("<|vision_start|><|image_pad|><|vision_end|>", response)
        self.assertLess(response.index("PDF TEXT HERE"), response.index("<|vision_start|>"))
        self.assertLess(response.index("<|vision_end|>"), response.index("tool text"))

    def test_a_screenshot_without_vision_is_a_note_not_an_error(self):
        from serve.frontend import images_of
        msgs, tool = self.tool_message(vision=False)
        self.assertEqual(images_of(msgs), [])
        self.assertIn("image", self.text(tool["content"]).lower())
        self.assertIn("not enabled", self.text(tool["content"]))


class ExitedProc:
    """#48 defect 2 (xeno): the engine process has exited (Windows showed HasExited = True) but its stdout never
    reached end-of-file, so no None ever arrived on the line queue."""
    returncode = 3221225477                             # 0xC0000005, an access violation

    def __init__(self, code=returncode):
        self.stdin, self.code = io.StringIO(), code

    def poll(self):
        return self.code

    def wait(self, timeout=None):
        return self.code


class QuietEngineLiveness(unittest.TestCase):
    """#49 S2 / #48 defect 2 (xeno): while the engine is quiet (reading a prompt), the server must notice that its
    process has ended; it used to print "reading the prompt" for minutes and Claude Code waited instead of retrying."""

    def engine(self, code):
        eng = StrataEngine.__new__(StrataEngine)
        eng.proc, eng.lines, eng.QUIET_S = ExitedProc(code), queue.Queue(), 0.01
        return eng

    def test_an_engine_that_exits_mid_prompt_ends_the_request(self):
        gen = self.engine(ExitedProc.returncode).generate([1, 2, 3], 10, {}, threading.Event())
        with self.assertRaises(EngineDied):
            for _ in zip(range(200), gen):
                pass

    def test_a_live_quiet_engine_keeps_sending_heartbeats_until_it_exits(self):
        eng = self.engine(None)
        gen = eng.generate([1, 2, 3], 10, {}, threading.Event())
        self.assertEqual([next(gen) for _ in range(3)], [None, None, None])
        eng.proc.code = ExitedProc.returncode
        with self.assertRaises(EngineDied):
            next(gen)
        self.assertFalse(eng.alive())                   # the next request starts it again (issue #27)


class EnginePipe(unittest.TestCase):
    """Review of #49 (xeno): the paths around the engine's pipe that the first S2 fix did not cover."""

    def engine(self, lines, code=None):
        eng = StrataEngine.__new__(StrataEngine)
        eng.proc, eng.lines, eng.QUIET_S, eng.can_stop = ExitedProc(code), queue.Queue(), 0.01, True
        for line in lines:
            eng.lines.put(line)
        return eng

    def test_an_engine_that_dies_during_the_stop_drain_does_not_hang(self):
        # #48's pipe that stays open, met in the STOP-and-drain after an early stop (a cancel, a stop token, the
        # budget cut): the drain waited on the queue for ever while holding the engine slot
        eng = self.engine(["T 5\n"])
        gen = eng.generate([1, 2], 10, {}, threading.Event())
        self.assertEqual(next(gen), 5)
        eng.proc.code = ExitedProc.returncode
        closer = threading.Thread(target=gen.close, daemon=True)
        closer.start()
        closer.join(5)
        self.assertFalse(closer.is_alive(), "the drain hangs on a dead engine")
        self.assertFalse(eng.alive())

    def test_tokens_drained_after_stop_are_kept(self):
        # the engine commits tokens ahead of what the server has read; a continuation must carry them to reuse
        # the engine's live session (its prefix match is on every committed token)
        eng = self.engine(["T 5\n", "T 6\n", "T 7\n", "DONE 3 2 1.0 1.0 stop\n"])
        gen = eng.generate([1, 2], 10, {}, threading.Event())
        self.assertEqual(next(gen), 5)
        gen.close()
        self.assertEqual(eng.drained, [6, 7])
        self.assertEqual(eng.last["generated"], 3)

    def test_a_pump_from_before_a_restart_leaves_the_new_engine_alone(self):
        # restart() replaces the process and the queue; the old pump, blocked on the old pipe until it closes,
        # used to mark the NEW engine ended and put None into the NEW queue - a spurious death and restart
        eng = StrataEngine.__new__(StrataEngine)
        old_proc, old_lines = ExitedProc(1), queue.Queue()
        old_proc.stdout = iter(["INFO x\n"])
        eng.proc, eng.lines, eng.ended = ExitedProc(None), queue.Queue(), False
        eng._pump(old_proc, old_lines)
        self.assertFalse(eng.ended)
        self.assertTrue(eng.lines.empty())
        self.assertEqual([old_lines.get_nowait(), old_lines.get_nowait()], ["INFO x\n", None])


class LegLog(MockEngine):
    """A mock with the real engine's two habits the thinking budget meets: after an early stop it has committed a
    few more tokens than were read (`drained`), and each call leaves its own DONE figures in `last`."""
    LEGS = [{"prompt_ms": 5000.0, "decode_ms": 20000.0, "generated": 103, "reused": 0, "drafts_accepted": 60,
             "drafts_offered": 80},
            {"prompt_ms": 40.0, "decode_ms": 200.0, "generated": 10, "reused": 374, "drafts_accepted": 5,
             "drafts_offered": 8}]

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts = getattr(self, "prompts", []) + [list(ids)]
        leg = len(self.prompts) - 1
        n = 0
        try:
            for t in super().generate(ids, max_new, sampling, cancel, embeddings):
                n += 1
                yield t
        finally:
            if leg == 0:
                self.drained = list(self.script[n:n + 3])
            self.last = dict(LegLog.LEGS[min(leg, 1)], prompt_tokens=len(ids))


class BudgetContinuation(unittest.TestCase):
    """Review of #49 S3 (xeno): the continuation after a thinking-budget cut."""

    def run_cut(self):
        tok = ByteTokenizer()
        eng = LegLog(tok, [THINK[:600], "the answer"], max_context=16384)
        svc = Service(eng, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        ids, thinking, max_new = svc.prepare([{"role": "user", "content": "hi"}], None, {}, 4000)
        out = list(svc.run(ids, thinking, None, max_new, {"_think_budget": 100}, threading.Event()))
        return tok, eng, svc, ids, out

    def test_the_continuation_carries_the_tokens_the_engine_wrote_past_the_cut(self):
        tok, eng, svc, ids, out = self.run_cut()
        first, second = eng.prompts
        self.assertEqual(tok.decode(second[len(first):len(first) + 103]), THINK[:103])
        self.assertTrue(tok.decode(second[len(first) + 103:]).startswith("\n\nI have thought enough"))
        shown = "".join(x.text for k, x in out if k == "event" and x.kind == "reasoning")
        self.assertTrue(shown.startswith(THINK[:103]))

    def test_the_numbers_cover_both_legs(self):
        tok, eng, svc, ids, out = self.run_cut()
        h = svc.history[-1]
        self.assertEqual(h["prompt_ms"], 5000.0)        # the real prompt read is leg 1's
        self.assertEqual(h["reused"], 0)                 # not leg 2's reuse of its own continuation
        self.assertEqual(h["decode_ms"], 20000.0 + 40.0 + 200.0)   # re-reading CLOSE is part of making the output


class RestartOverloaded(unittest.TestCase):
    """Review of #49 S2 (xeno): '529 overloaded_error while the engine is down or LOADING' - a failed restart was an
    uncaught RuntimeError (a dropped connection), and a request during a restart waited minutes for it."""

    def serve_with(self, eng):
        tok = ByteTokenizer()
        svc = Service(eng, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        base = f"http://127.0.0.1:{httpd.server_address[1]}/v1/messages"

        def post(timeout=30):
            body = {"model": "m", "max_tokens": 20, "messages": [{"role": "user", "content": "hi"}]}
            req = urllib.request.Request(base, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    return r.status, r.read().decode()
            except urllib.error.HTTPError as e:
                with e:
                    return e.code, e.read().decode()
        return svc, post

    def test_a_failed_restart_is_529(self):
        class FailsToRestart(MockEngine):
            def alive(self):
                return False

            def restart(self):
                raise RuntimeError("the engine exited before it was ready")

        _, post = self.serve_with(FailsToRestart(ByteTokenizer(), "</think>\n\nok", max_context=CTX))
        status, text = post()
        self.assertEqual(status, 529, text)
        self.assertEqual(json.loads(text)["error"]["type"], "overloaded_error")

    def test_a_request_during_a_restart_is_529_at_once(self):
        gate = threading.Event()

        class SlowRestart(MockEngine):
            dead = True

            def alive(self):
                return not self.dead

            def restart(self):
                gate.wait(20)
                self.dead = False

        svc, post = self.serve_with(SlowRestart(ByteTokenizer(), "</think>\n\nok", max_context=CTX))
        first = threading.Thread(target=post, daemon=True)
        first.start()
        for _ in range(500):
            if svc.restarting:
                break
            time.sleep(0.01)
        t0 = time.time()
        status, text = post(timeout=10)
        gate.set()
        first.join(20)
        self.assertEqual(status, 529, text)
        self.assertLess(time.time() - t0, 5)


class DeadAtOnce(MockEngine):
    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        raise EngineDied("the engine stopped unexpectedly (exit code 3221225477)")
        yield                                           # a generator, as the real one is


class AnthropicOverloaded(unittest.TestCase):
    """#49 S2 (xeno): a dead engine is the native API's `529 overloaded_error` on /v1/messages - the error Claude Code
    retries with backoff (it was 503 server_error / an api_error event, which it shows as a failed turn)."""

    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.svc = Service(DeadAtOnce(tok, ANSWER, max_context=CTX), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, body):
        req = urllib.request.Request(self.base + "/v1/messages", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, r.read().decode()
        except urllib.error.HTTPError as e:
            with e:
                return e.code, e.read().decode()

    BODY = {"model": "m", "max_tokens": 20, "messages": [{"role": "user", "content": "hi"}]}

    def test_not_streamed(self):
        status, text = self.post(self.BODY)
        self.assertEqual(status, 529, text)
        body = json.loads(text)
        self.assertEqual(body["type"], "error")
        self.assertEqual(body["error"]["type"], "overloaded_error")

    def test_streamed(self):
        status, text = self.post({**self.BODY, "stream": True})
        self.assertEqual(status, 200)
        self.assertIn("event: error", text)
        err = json.loads(text.split("event: error\ndata: ", 1)[1].split("\n", 1)[0])
        self.assertEqual(err["error"]["type"], "overloaded_error")


def _call(name):
    return f"<tool_call>\n<function={name}>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>"


class ParallelToolUse(unittest.TestCase):
    """#49 S2 (xeno): tool_choice.disable_parallel_tool_use - the native API returns at most one tool_use."""

    TOOLS = [{"name": n, "description": "d", "input_schema": {"type": "object", "properties": {"x": {"type": "string"}}}}
             for n in ("first", "second")]

    def uses(self, tool_choice):
        tok = ByteTokenizer()
        eng = MockEngine(tok, "</think>\n\n" + _call("first") + "\n" + _call("second"), max_context=CTX)
        svc = Service(eng, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        try:
            body = {"model": "m", "max_tokens": 400, "tools": self.TOOLS,
                    "messages": [{"role": "user", "content": "go"}], **tool_choice}
            req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}/v1/messages",
                                         data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                out = json.loads(r.read())
        finally:
            httpd.shutdown()
            httpd.server_close()
        self.history = svc.history
        return [b["name"] for b in out["content"] if b["type"] == "tool_use"], out["stop_reason"]

    def test_parallel_by_default(self):
        self.assertEqual(self.uses({}), (["first", "second"], "tool_use"))

    def test_one_tool_use_when_disabled(self):
        choice = {"tool_choice": {"type": "auto", "disable_parallel_tool_use": True}}
        self.assertEqual(self.uses(choice), (["first"], "tool_use"))
        # the server's own stop is not a client cancel (review of #49): the engine stopped early, finish "stop"
        self.assertEqual(self.history[-1]["finish"], "stop")
        self.assertLess(self.history[-1]["output_tokens"], len("</think>\n\n" + _call("first") + "\n" + _call("second")))


THINK = "".join("abcdefghijklmnopqrstuvwxyz"[(i * 7919 + i * i * 104729) % 26] for i in range(4000))


class PromptLog(MockEngine):
    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts = getattr(self, "prompts", []) + [list(ids)]
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class ThinkingBudget(unittest.TestCase):
    """#49 S3 (xeno): Anthropic's thinking.budget_tokens closes the thinking block once spent (it only picked an
    effort level), and a non-streamed side request - Claude Code's auto-mode classifier - thinks little: it held the
    only slot for 30-130 s on the EXL3 server while the main turn queued."""

    def run_request(self, scripts, body, max_context=16384):
        tok = ByteTokenizer()
        eng = PromptLog(tok, scripts, max_context=max_context)
        svc = Service(eng, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        try:
            body = {"model": "m", "max_tokens": 4000, "messages": [{"role": "user", "content": "hi"}], **body}
            req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}/v1/messages",
                                         data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read().decode()
        finally:
            httpd.shutdown()
            httpd.server_close()
        if body.get("stream"):
            thinking, text = "", ""
            for line in raw.splitlines():
                if line.startswith("data: "):
                    d = json.loads(line[6:]).get("delta") or {}
                    thinking += d.get("thinking", "")
                    text += d.get("text", "")
        else:
            out = json.loads(raw)
            thinking = "".join(b.get("thinking", "") for b in out["content"] if b["type"] == "thinking")
            text = "".join(b.get("text", "") for b in out["content"] if b["type"] == "text")
        return tok, eng, thinking, text

    def test_the_budget_closes_thinking_and_the_answer_follows(self):
        tok, eng, thinking, text = self.run_request(
            [THINK[:600], "the answer"], {"stream": True, "thinking": {"type": "enabled", "budget_tokens": 100}})
        self.assertEqual(thinking[:thinking.index("\n")], THINK[:100])    # cut at the budget, then the close
        self.assertEqual(text, "the answer")
        first, second = eng.prompts                     # the answer continues the cut prompt, closed by the server
        self.assertEqual(second[:len(first)], first)
        self.assertEqual(tok.decode(second[len(first):len(first) + 100]), THINK[:100])
        self.assertTrue(tok.decode(second[len(first) + 100:]).endswith("</think>\n\n"))

    def test_a_budget_that_runs_out_after_the_model_closed_thinking_does_nothing(self):
        # review of #49: the cut keyed on parser EVENTS, and '</think>', the newlines after it and a held
        # '<tool_call>' prefix emit none - so a budget spent there injected CLOSE into the answer or the call
        for tail in ("</think>\n\nthe answer",
                     "</think>\n\n<tool_call>\n<function=Read>\n<parameter=file_path>\n/x\n</parameter>\n</function>\n"
                     "</tool_call>"):
            for extra in (1, 3, 9):             # the budget lands this many tokens after '</think>' ends
                budget = 92 + len("</think>") + extra
                tok, eng, thinking, text = self.run_request(
                    [THINK[:92] + tail, "LEG TWO"],
                    {"stream": True, "thinking": {"type": "enabled", "budget_tokens": budget},
                     "tools": [{"name": "Read", "description": "d", "input_schema": {"type": "object"}}]})
                self.assertEqual(len(eng.prompts), 1, (tail[:20], extra))
                self.assertNotIn("I have thought enough", thinking + text)
                self.assertNotIn("LEG TWO", text)

    def test_the_budget_is_capped_by_the_max_tokens_the_engine_really_gets(self):
        # review of #49: the budget was capped against the raw max_tokens before prepare() - 0 means "the rest of
        # the context", so a 31,999 budget never fired and the turn thought to the end with no answer
        tok, eng, thinking, text = self.run_request(
            [THINK[:3900], "the answer"], {"stream": True, "max_tokens": 0,
                                           "thinking": {"type": "enabled", "budget_tokens": 31999}}, max_context=6000)
        self.assertEqual(len(eng.prompts), 2)
        self.assertEqual(text, "the answer")

    def test_a_streamed_request_without_a_budget_thinks_freely(self):
        _, eng, thinking, text = self.run_request([THINK[:1500] + "</think>\n\nok"], {"stream": True})
        self.assertEqual(thinking.strip(), THINK[:1500])
        self.assertEqual(text, "ok")
        self.assertEqual(len(eng.prompts), 1)

    def test_a_side_request_thinks_little(self):
        tok, eng, thinking, text = self.run_request([THINK[:3000], "no"], {})
        self.assertEqual(thinking[:thinking.index("\n")], THINK[:1024])
        self.assertEqual(text, "no")
        self.assertIn("Reasoning effort is set to low", tok.decode(eng.prompts[0]))

    def test_a_main_turn_resent_without_streaming_thinks_freely(self):
        """Live (2026-09-30): after a failed stream Claude Code sent its main turn again without streaming (102,165
        tokens, 33 tools), and it was taken for a side request: its thinking was closed at 1,024 tokens. The side
        requests (the classifier, titles) carry no tools; a request with tools is a main turn."""
        tool = {"name": "Read", "description": "read a file", "input_schema": {"type": "object", "properties": {}}}
        tok, eng, thinking, text = self.run_request([THINK[:3000] + "</think>\n\nok"], {"tools": [tool]})
        self.assertEqual(thinking.strip(), THINK[:3000])
        self.assertEqual(len(eng.prompts), 1)
        self.assertNotIn("Reasoning effort is set to low", tok.decode(eng.prompts[0]))

    def test_side_budget_zero_leaves_side_requests_alone(self):
        from unittest import mock
        with mock.patch.dict(os.environ, {"STRATA_SIDE_BUDGET": "0"}):
            tok, eng, thinking, _ = self.run_request([THINK[:3000] + "</think>\n\nno"], {})
        self.assertEqual(thinking.strip(), THINK[:3000])
        self.assertNotIn("Reasoning effort is set to low", tok.decode(eng.prompts[0]))


class TimingLine(unittest.TestCase):
    """#49 S5 (xeno): the end-of-request timing in llama-server's shape, which our tools and eyes already read."""

    LAST = {"prompt_ms": 500.0, "decode_ms": 1000.0, "generated": 50, "reused": 900,
            "drafts_accepted": 30, "drafts_offered": 40}

    def test_the_block(self):
        from serve.timing_line import report
        lines = report(self.LAST, prompt_tokens=1000, wall_s=1.75)
        self.assertEqual(lines[0], "prompt eval time =     500.00 ms /   100 tokens (    5.00 ms per token,   "
                                   "200.00 tokens per second)  [900 cached]")
        self.assertEqual(lines[1], "       eval time =    1000.00 ms /    50 tokens (   20.00 ms per token,    "
                                   "50.00 tokens per second)")
        self.assertEqual(lines[2], "      total time =    1750.00 ms /   150 tokens")
        self.assertEqual(lines[3], "draft acceptance = 0.75000 (   30 accepted /    40 generated)")

    def test_a_request_prints_it(self):
        class Timed(MockEngine):
            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                self.last = dict(TimingLine.LAST, prompt_tokens=len(ids))   # the real one: at DONE, which the
                yield from super().generate(ids, max_new, sampling, cancel, embeddings)   # stop-token drain reads

        import contextlib
        tok = ByteTokenizer()
        svc = Service(Timed(tok, "</think>\n\nok", max_context=CTX), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        ids, thinking, max_new = svc.prepare([{"role": "user", "content": "hi"}], None, {}, 20)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            list(svc.run(ids, thinking, None, max_new, {}, threading.Event()))
        self.assertIn("prompt eval time =     500.00 ms", out.getvalue())
        self.assertIn("draft acceptance = 0.75000", out.getvalue())


class SseTrace(unittest.TestCase):
    """#49 S5 (xeno): STRATA_TRACE_SSE=<file> records every SSE event sent on /v1/messages, one JSON line each."""

    def test_trace(self):
        import tempfile
        from unittest import mock
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "</think>\n\nok", max_context=CTX), tok,
                      ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "trace.jsonl"
            try:
                with mock.patch.dict(os.environ, {"STRATA_TRACE_SSE": str(path)}):
                    body = {"model": "m", "max_tokens": 20, "stream": True,
                            "messages": [{"role": "user", "content": "hi"}]}
                    req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}/v1/messages",
                                                 data=json.dumps(body).encode(),
                                                 headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=30) as r:
                        sent = [ln[7:] for ln in r.read().decode().splitlines() if ln.startswith("event: ")]
            finally:
                httpd.shutdown()
                httpd.server_close()
            recs = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([r["ev"] for r in recs], sent)
        self.assertEqual(sent[0], "message_start")
        self.assertEqual(sent[-1], "message_stop")
        self.assertEqual(len({r["rid"] for r in recs}), 1)
        self.assertTrue(all(isinstance(r["t"], float) for r in recs))
        self.assertEqual("".join((r.get("delta") or {}).get("text", "") for r in recs), "ok")


class Priority(unittest.TestCase):
    """#49 S6 (xeno): Claude Code's main turn streams; its title and auto-mode classifier requests do not. A queued
    streamed request goes before queued non-streamed ones; the running request is never pre-empted."""

    def test_a_streamed_request_goes_before_waiting_side_requests(self):
        class Gated(MockEngine):
            order, gate = [], threading.Event()

            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                Gated.order.append(sampling.get("tag"))
                if sampling.get("tag") == "running":
                    Gated.gate.wait(10)
                yield from super().generate(ids, max_new, sampling, cancel, embeddings)

        tok = ByteTokenizer()
        svc = Service(Gated(tok, "</think>\n\nok", max_context=CTX), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        ids, thinking, max_new = svc.prepare([{"role": "user", "content": "hi"}], None, {}, 20)

        def request(tag, stream):
            list(svc.run(ids, thinking, None, max_new, {"tag": tag, "stream": stream}, threading.Event()))

        def queued(k):
            for _ in range(500):
                if svc.status["queued"] >= k:
                    return
                time.sleep(0.01)
            self.fail(f"{k} requests never queued")

        threads = [threading.Thread(target=request, args=("running", True))]
        threads[0].start()
        while not Gated.order:
            time.sleep(0.01)
        for i, (tag, stream) in enumerate([("side-1", False), ("side-2", False), ("main", True)]):
            threads.append(threading.Thread(target=request, args=(tag, stream)))
            threads[-1].start()
            queued(i + 1)
        Gated.gate.set()
        for t in threads:
            t.join(10)
        self.assertEqual(Gated.order, ["running", "main", "side-1", "side-2"])


class ImagePriority(unittest.TestCase):
    """Review of #49 S6 (xeno): encoding a request's images takes the engine slot too; a main turn with a screenshot
    in its history encoded at side-request priority, behind every queued classifier request."""

    def test_a_main_turn_encodes_its_images_before_waiting_side_requests(self):
        import tempfile
        order, gate = [], threading.Event()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)

        class FakeVision:
            dir = Path(tmp.name)

            def encode(self, src):
                order.append("encode")
                f = Path(tmp.name) / "img.bin"
                f.write_bytes(b"x")
                return f, 1

        class Gated(MockEngine):
            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                order.append(sampling.get("tag"))
                if sampling.get("tag") == "running":
                    gate.wait(10)
                yield from super().generate(ids, max_new, sampling, cancel, embeddings)

        tok = ByteTokenizer()
        svc = Service(Gated(tok, "</think>\n\nok", max_context=CTX), tok,
                      ChatTemplate(ROOT / "serve/chat_template.jinja"), vision=FakeVision())
        ids, thinking, max_new = svc.prepare([{"role": "user", "content": "hi"}], None, {}, 20)

        def run(tag, stream):
            list(svc.run(ids, thinking, None, max_new, {"tag": tag, "stream": stream}, threading.Event()))

        threads = [threading.Thread(target=run, args=("running", True), daemon=True)]
        threads[0].start()
        while "running" not in order:
            time.sleep(0.01)
        threads.append(threading.Thread(target=run, args=("side", False), daemon=True))
        threads[-1].start()
        while svc.status["queued"] < 1:
            time.sleep(0.01)
        image = [{"role": "user", "content": [{"type": "image", "source": "data:image/png;base64,AAAA"},
                                              {"type": "text", "text": "look"}]}]
        threads.append(threading.Thread(target=svc.prepare, args=(image, None, {}, 20), kwargs={"priority": 0},
                                        daemon=True))
        threads[-1].start()
        time.sleep(0.3)                                 # the encode is now waiting for the slot
        gate.set()
        for t in threads:
            t.join(10)
        self.assertEqual(order, ["running", "encode", "side"])


class SamplingPresets(unittest.TestCase):
    """#49 S8 / story 20 (xeno): coding presets (deterministic, balanced, reasoning) chosen in the run config's
    sampling block - never switched automatically; the block's own keys override the preset's."""

    def defaults(self, block):
        from serve.server import sampling_defaults_from_config
        return sampling_defaults_from_config({"sampling": block})

    def test_presets(self):
        self.assertEqual(self.defaults({"preset": "deterministic"}), {"temperature": 0.0})
        self.assertEqual(self.defaults({"preset": "balanced"}), {"temperature": 0.6, "top_p": 0.95, "top_k": 20})
        self.assertEqual(self.defaults({"preset": "reasoning"}), {"temperature": 1.0, "top_p": 0.95, "top_k": 20})

    def test_own_keys_win(self):
        self.assertEqual(self.defaults({"preset": "reasoning", "top_k": 40}),
                         {"temperature": 1.0, "top_p": 0.95, "top_k": 40})

    def test_no_preset_is_unchanged(self):
        self.assertEqual(self.defaults({"temperature": 0.7}), {"temperature": 0.7})

    def test_an_unknown_preset_refuses_to_start(self):
        with self.assertRaises(SystemExit):
            self.defaults({"preset": "creative"})


class LogEncoding(unittest.TestCase):
    """2026-09-30, live (xeno): with STRATA_DEBUG=1 the raw-output line went to a redirected stdout in cp1252; a Thai
    answer raised UnicodeEncodeError - a ValueError - inside the request, the SSE loop sent an `error` event instead of
    message_stop, and Claude Code showed "Server error mid-response" on 3 of 3 Thai tool-call turns."""

    def test_a_log_line_the_console_cannot_encode_never_fails_the_request(self):
        import contextlib
        from unittest import mock
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "</think>\n\nคำตอบภาษาไทย", max_context=CTX), tok,
                      ChatTemplate(ROOT / "serve/chat_template.jinja"))
        ids, thinking, max_new = svc.prepare([{"role": "user", "content": "hi"}], None, {}, 64)
        cp1252 = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")        # as a redirected stdout on Windows
        with mock.patch.dict(os.environ, {"STRATA_DEBUG": "1"}), contextlib.redirect_stdout(cp1252):
            out = list(svc.run(ids, thinking, None, max_new, {}, threading.Event()))
        self.assertEqual(out[-1][0], "done")
        cp1252.flush()
        self.assertIn(b"[strata] raw:", cp1252.buffer.getvalue())


class CjkGuard(unittest.TestCase):
    """#49 S4 (xeno; the EXL3 server's #77): Han tokens are banned for a request whose prompt has no Han character
    and does not name Chinese - 4.0bpw dropped Han characters into Thai sentences by sampling drift."""

    def msgs(self, text):
        return [{"role": "user", "content": text}]

    def test_wanted(self):
        from serve.cjk_guard import wanted
        self.assertTrue(wanted(self.msgs("ช่วยสรุปไฟล์นี้หน่อย")))
        self.assertTrue(wanted(self.msgs("explain these machinations")))       # word-bounded: not "china"
        self.assertFalse(wanted(self.msgs("翻译这个")))
        self.assertFalse(wanted(self.msgs("แปลเป็นภาษาจีน")))
        self.assertFalse(wanted(self.msgs("reply in Chinese")))
        self.assertFalse(wanted([{"role": "tool", "content": [{"type": "text", "text": "注释"}]}]))
        from unittest import mock
        with mock.patch.dict(os.environ, {"STRATA_ALLOW_CJK": "1"}):
            self.assertFalse(wanted(self.msgs("hello")))

    def test_only_the_current_turn_decides(self):
        # review of #49: the whole conversation was scanned - the system prompt (CLAUDE.md, the memory index says
        # "no Chinese unless asked") or one earlier Read of a file with Han lifted the ban for good
        from serve.cjk_guard import wanted
        convo = [{"role": "system", "content": "EXL3: no Chinese unless asked"},
                 {"role": "user", "content": "อ่านไฟล์นี้"},
                 {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "Read", "arguments": {}}}]},
                 {"role": "tool", "content": "# 注释 a comment"},
                 {"role": "assistant", "content": "สรุปแล้ว"},
                 {"role": "user", "content": "ต่อเลย"}]
        self.assertTrue(wanted(convo))
        self.assertFalse(wanted(convo[:4]))             # the current turn IS the tool result with Han
        self.assertFalse(wanted(convo[:-1] + [{"role": "user", "content": "แปลเป็นภาษาจีน"}]))

    def test_ban_ids(self):
        from serve.cjk_guard import ban_ids
        pieces = ["a", "中", " 文字", "ก", "�", "の"]         # kana and a broken byte are not Han
        self.assertEqual(ban_ids(lambda i: pieces[i], len(pieces)), [1, 2])

    def test_the_gen_key(self):
        self.assertIn("ban=1", StrataEngine.sampling_keys({"_ban": True}).split())
        self.assertFalse([k for k in StrataEngine.sampling_keys({}).split() if k.startswith("ban=")])

    def test_an_image_request_carries_the_ban(self):
        eng = StrataEngine.__new__(StrataEngine)
        eng.proc, eng.lines, eng.QUIET_S = ExitedProc(None), queue.Queue(), 0.01
        eng.lines.put("DONE 0 1 0 0 stop\n")
        list(eng.generate([1, 2], 5, {"_ban": True}, threading.Event(), embeddings="req.sve"))
        head = eng.proc.stdin.getvalue().split(" 1,2")[0]
        self.assertTrue(head.startswith("GENI 5"), head)
        self.assertIn("ban=1", head.split())

    def test_the_server_bans_per_request(self):
        class Recorder(MockEngine):
            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                self.bans = getattr(self, "bans", []) + [bool(sampling.get("_ban"))]
                yield from super().generate(ids, max_new, sampling, cancel, embeddings)

        tok = ByteTokenizer()
        eng = Recorder(tok, "</think>\n\nok 中文", max_context=CTX)        # a leak, as the instrument sees it
        svc = Service(eng, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        try:
            def post(path, text):
                body = {"model": "m", "max_tokens": 20, "stream": True, "messages": self.msgs(text)}
                req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}{path}",
                                             data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    r.read()
            post("/v1/messages", "สวัสดี")                   # the engine has no ban list: never sent
            svc.cjk_ban = True
            post("/v1/messages", "สวัสดี")
            post("/v1/messages", "翻译")
            post("/v1/chat/completions", "hello")
        finally:
            httpd.shutdown()
            httpd.server_close()
        self.assertEqual(eng.bans, [False, True, False, True])
        self.assertEqual([r["cjk_chars"] for r in svc.metrics()["requests"]], [2, 2, 2, 2])


class CacheSlotProtocol(unittest.TestCase):
    def test_slot_selection_and_validation(self):
        self.assertIn("cache_slot=0", StrataEngine.sampling_keys({}))
        for slot in range(4):
            self.assertIn(f"cache_slot={slot}", StrataEngine.sampling_keys({"strata_cache_slot":slot}))
        for slot in [-1,4,True,"1",1.5]:
            with self.assertRaises(ValueError):
                StrataEngine.sampling_keys({"strata_cache_slot":slot})


class AutoCacheSlot(unittest.TestCase):
    """#49 S7 (xeno, on upstream PR #175's slots): Claude Code cannot send strata_cache_slot, and its side requests
    (the auto-mode classifier, titles) have their own prompt, so each one wiped the main session's cache - a 52.6K
    re-read (55 s) in real use (#50). The server picks a slot per prompt family (the prompt's first tokens), keeps a
    family on its slot, and reuses the least recently used slot when there are more families than slots."""

    class SlotEngine(MockEngine):
        def generate(self, ids, max_new, sampling, cancel, embeddings=None):
            self.slots = getattr(self, "slots", []) + [sampling.get("strata_cache_slot", 0)]
            self.samplings = getattr(self, "samplings", []) + [dict(sampling)]
            yield from super().generate(ids, max_new, sampling, cancel, embeddings)

    def serve_with(self, slots):
        tok = ByteTokenizer()
        eng = self.SlotEngine(tok, "</think>\n\nok", max_context=16384)
        if slots is not None:
            eng.info = {"cache_slots": slots}
        svc = Service(eng, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)

        def post(system, extra=None):
            body = {"model": "m", "max_tokens": 8, "stream": True, "system": system,
                    "messages": [{"role": "user", "content": "hi"}], **(extra or {})}
            req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}/v1/messages",
                                         data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                r.read()
        return eng, post

    MAIN, SIDE = "You are the main agent. " * 40, "You are a classifier. " * 40

    def test_families_keep_their_own_slot(self):
        eng, post = self.serve_with(4)
        for system in (self.MAIN, self.SIDE, self.MAIN, self.SIDE, self.MAIN):
            post(system)
        self.assertEqual(eng.slots[0], eng.slots[2])
        self.assertEqual(eng.slots[0], eng.slots[4])
        self.assertEqual(eng.slots[1], eng.slots[3])
        self.assertNotEqual(eng.slots[0], eng.slots[1])

    def test_the_least_recently_used_family_gives_up_its_slot(self):
        eng, post = self.serve_with(2)
        post("family A " * 80)
        post("family B " * 80)
        post("family A " * 80)          # A is now the more recent
        post("family C " * 80)          # takes B's slot, not A's
        self.assertEqual(eng.slots[3], eng.slots[1])
        self.assertNotEqual(eng.slots[3], eng.slots[0])

    def test_an_explicit_slot_wins(self):
        eng, post = self.serve_with(4)
        post(self.MAIN, {"strata_cache_slot": 3})
        self.assertEqual(eng.slots, [3])

    def test_a_family_reports_where_its_prompt_leaves_the_last_one(self):
        """#49 S7 follow-up: the classifier's stage 2 reused only the 28,408-token root of its stage 1, though both
        render the same transcript. Where the two prompts part is the evidence for where a checkpoint would help."""
        from serve.server import CacheSlots
        slots = CacheSlots(4)
        head = list(range(600))
        slots.pick(head + [1, 2, 3])
        self.assertEqual(slots.shared(head + [1, 9, 9, 9]), 601)
        self.assertIsNone(slots.shared([7] * 600))      # a family never seen: nothing to compare with

    def test_the_engine_checkpoints_where_the_family_last_parted(self):
        """Live (2026-09-30, engine-s7.log): a classifier prompt of 39,994 tokens shared 36,158 with the one before
        it but reused only its 28,408-token root - its transcript grows inside one message, and the engine keeps
        checkpoints at turn starts only. The server names the shared length (GEN key ckpt_at) so the engine keeps a
        checkpoint there, which the family's next prompt shares too."""
        from serve.server import StrataEngine
        eng, post = self.serve_with(4)
        post(self.SIDE)
        self.assertNotIn("_ckpt_at", eng.samplings[0])                # nothing to compare with yet
        post(self.SIDE + " and more")
        shared = eng.samplings[1]["_ckpt_at"]
        self.assertGreater(shared, len(self.SIDE))
        self.assertIn(f" ckpt_at={shared}", StrataEngine.sampling_keys(eng.samplings[1]))
        self.assertNotIn("ckpt_at", StrataEngine.sampling_keys(eng.samplings[0]))

    def test_no_automatic_slot_without_engine_slots(self):
        for slots in (None, 1):
            eng, post = self.serve_with(slots)
            post(self.MAIN)
            post(self.SIDE)
            self.assertEqual(eng.slots, [0, 0], slots)


class ClassifierStageOne(unittest.TestCase):
    """#49 S7 follow-up (xeno, decided by the developer 2026-09-30): Claude Code's auto-mode classifier asks its fast
    stage for a reply that MUST begin with <block>, reads it with `<block>(yes|no)`, and gives it 64 tokens without
    thinking. Qwen answered in prose ("Evaluating the final action: ..."), was cut at 64 tokens, and the stage was
    sent again 4 times with the same greedy answer before the slow stage ran - every round, in real use.
    A: the server writes the required opening for the model. B: an identical greedy request is answered again from
    the last answer instead of being generated again."""

    MUST = "Err on the side of blocking. Your ENTIRE response MUST begin with <block>. Do NOT output anything else."

    class Counting(MockEngine):
        def generate(self, ids, max_new, sampling, cancel, embeddings=None):
            self.calls = getattr(self, "calls", 0) + 1
            yield from super().generate(ids, max_new, sampling, cancel, embeddings)

    def serve_with(self, script):
        tok = ByteTokenizer()
        eng = self.Counting(tok, script, max_context=16384)
        svc = Service(eng, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)

        def post(last, system="You judge actions.", extra=None):
            body = {"model": "m", "max_tokens": 64, "stream": False, "system": system,
                    "thinking": {"type": "disabled"},        # as the classifier sends it (thinking=False in the log)
                    "messages": [{"role": "user", "content": last}], **(extra or {})}
            req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}/v1/messages",
                                         data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        return tok, eng, post

    @staticmethod
    def text(body):
        return "".join(b.get("text", "") for b in body["content"] if b["type"] == "text")

    def test_A_the_required_opening_is_written_for_the_model(self):
        tok, eng, post = self.serve_with("no</block>")
        body = post("<transcript>ls</transcript>\n" + self.MUST)
        self.assertEqual(self.text(body), "<block>no</block>")
        self.assertEqual(eng.last_prompt[-len("<block>"):], tok.encode("<block>"))   # the model continues it

    def test_A_no_opening_without_the_rule_in_the_last_message(self):
        tok, eng, post = self.serve_with("fine")
        self.assertEqual(self.text(post("<transcript>ls</transcript>")), "fine")
        self.assertEqual(self.text(post("<transcript>ls</transcript>", system=self.MUST,      # stage 2: it thinks
                                        extra={"max_tokens": 8192})), "fine")               # before its <block>

    # the fast stage's own last lines in real use (logged with the developer's approval, 2026-09-30)
    STAGE1 = ("</transcript>\n\nStage 1 does NOT apply user intent or ALLOW exceptions — stage 2 will handle "
              "those.\nRespond with <severity>N</severity> ONLY. Grade HARM ONLY — do NOT reduce for user "
              "intent. No other text.\n")

    def test_A_the_real_fast_stage_gets_its_severity_opening(self):
        """Live (2026-09-30): the real fast stage asks for "<severity>N</severity> ONLY" (not <block>); Qwen wrote
        "The agent's most recent action is ..." instead."""
        tok, eng, post = self.serve_with("12</severity>")
        self.assertEqual(self.text(post("<transcript>\n{\"Bash\":\"ls\"}\n" + self.STAGE1)), "<severity>12</severity>")

    def test_A_a_format_only_in_the_system_prompt_is_not_forced(self):
        tok, eng, post = self.serve_with("fine")
        self.assertEqual(self.text(post("<transcript>ls</transcript>", system=self.STAGE1)), "fine")

    def test_B_an_identical_greedy_request_is_answered_again(self):
        tok, eng, post = self.serve_with("Evaluating the final action")
        first = post("<transcript>ls</transcript>\n" + self.MUST)
        again = post("<transcript>ls</transcript>\n" + self.MUST)
        self.assertEqual(eng.calls, 1)
        self.assertEqual(again["content"], first["content"])
        self.assertNotEqual(again["id"], first["id"])

    def test_B_sampled_different_or_ordinary_requests_are_generated(self):
        tok, eng, post = self.serve_with("x")
        post("<transcript>ls</transcript>\n" + self.MUST, extra={"temperature": 0.7})
        post("<transcript>ls</transcript>\n" + self.MUST, extra={"temperature": 0.7})
        post("<transcript>ls -la</transcript>\n" + self.MUST)
        post("<transcript>ls</transcript>\n" + self.MUST)
        post("the same benchmark prompt")                # a repeated prompt is timed, never replayed
        post("the same benchmark prompt")
        self.assertEqual(eng.calls, 6)


class LoadingAnswers529(unittest.TestCase):
    """Live (2026-09-30): the server bound its port only after the model loaded (1-2 minutes), so every restart
    refused Claude Code's connections; it took them for a network fault and backed off until "will retry in 29m".
    EXL3 (C:\\AI a04b381) answers 529 overloaded_error while loading, which Claude Code retries soon. So does this."""

    def test_a_request_while_loading_is_529_overloaded_and_the_real_server_takes_the_port_after(self):
        from serve.server import loading_server
        placeholder = loading_server("127.0.0.1", 0)
        port = placeholder.server_address[1]
        body = json.dumps({"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/messages", data=body,
                                     headers={"Content-Type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(req, timeout=10)
        with caught.exception as e:
            self.assertEqual(e.code, 529)
            self.assertEqual(json.loads(e.read())["error"]["type"], "overloaded_error")
        with self.assertRaises(urllib.error.HTTPError) as health:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10)
        with health.exception as e:
            self.assertEqual(e.code, 503)
        placeholder.shutdown()
        placeholder.server_close()
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "</think>\n\nok"), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=port)                   # the same port, at once
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        with urllib.request.urlopen(req, timeout=10) as r:
            self.assertEqual(r.status, 200)

class UnloadableEngine(MockEngine):
    """A mock engine that can be stopped and started again like StrataEngine (alive / unload / restart)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.running, self.unloaded, self.starts = True, False, 0

    def alive(self):
        return self.running

    def unload(self):
        self.running, self.unloaded = False, True

    def restart(self):
        self.running, self.unloaded = True, False
        self.starts += 1


class SharingTheGpu(unittest.TestCase):
    """Idle unload, POST /unload and /load, the free-VRAM guard and the before_load hook (all off by default)."""

    def setUp(self):
        tok = ByteTokenizer()
        self.engine = UnloadableEngine(tok, "</think>\n\nok", max_context=CTX)
        self.svc = Service(self.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def req(self, path, body=None):
        r = urllib.request.Request(self.base + path, data=None if body is None else json.dumps(body).encode(),
                                   headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
        try:
            with urllib.request.urlopen(r, timeout=30) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def chat(self):
        return self.req("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}]})

    def test_unload_then_the_next_request_loads(self):
        self.assertEqual(self.req("/unload", {}), (200, {"status": "unloaded"}))
        self.assertFalse(self.engine.alive())
        self.assertEqual(self.req("/health")[1]["loaded"], False)
        self.assertEqual(self.req("/v1/models")[1]["data"][0]["status"]["value"], "unloaded")
        self.assertEqual(self.req("/unload", {}), (200, {"status": "not loaded"}))
        s, b = self.chat()
        self.assertEqual(s, 200)
        self.assertEqual(b["choices"][0]["message"]["content"], "ok")
        self.assertEqual(self.engine.starts, 1)
        self.assertEqual(self.req("/health")[1]["loaded"], True)

    def test_load_endpoint(self):
        self.svc.unload()
        self.assertEqual(self.req("/load", {}), (200, {"status": "loaded"}))
        self.assertTrue(self.engine.alive())
        self.assertEqual(self.req("/load", {}), (200, {"status": "loaded"}))
        self.assertEqual(self.engine.starts, 1)

    def test_unload_refused_while_a_request_runs(self):
        with self.svc.fifo:
            self.assertEqual(self.svc.unload(), "busy")
        self.assertTrue(self.engine.alive())

    def test_idle_unload(self):
        self.svc.idle_unload_s = 1
        self.svc.last_request_at = time.time()
        self.assertEqual(self.svc.unload(idle_for=1), "busy")       # a request just now: not idle yet
        self.svc.start_idle_unload()
        deadline = time.time() + 10
        while self.engine.alive() and time.time() < deadline:
            time.sleep(0.1)
        self.assertFalse(self.engine.alive())
        self.assertEqual(self.chat()[0], 200)

    def test_min_free_vram_refuses_to_load(self):
        self.svc.min_free_vram_mib = 8000
        self.svc.free_vram_mib = lambda: 2000
        self.svc.unload()
        t0 = time.time()
        s, b = self.chat()
        self.assertEqual(s, 503)
        self.assertIn("in use by another program", b["error"]["message"])
        self.assertFalse(self.engine.alive())
        self.assertGreater(time.time() - t0, 10)                    # waited for memory being given back first
        self.svc.free_vram_mib = lambda: 9000
        self.assertEqual(self.chat()[0], 200)

    def test_min_free_vram_unreadable_loads(self):
        self.svc.min_free_vram_mib = 8000
        self.svc.free_vram_mib = lambda: None                       # no NVML: never refuse
        self.svc.unload()
        self.assertEqual(self.chat()[0], 200)

    def test_before_load_runs_first(self):
        mark = Path(tempfile.mkdtemp()) / "ran"
        self.svc.before_load = [sys.executable, "-c", f"open({str(mark)!r}, 'w').close()"]
        self.svc.unload()
        self.assertFalse(mark.exists())
        self.assertEqual(self.chat()[0], 200)
        self.assertTrue(mark.exists())

    def test_vision_encoder_unloads_and_starts_first(self):
        order = []

        class FakeVision:
            running = True

            def alive(self):
                return self.running

            def unload(self):
                self.running = False

            def restart(self):
                order.append("vision")
                self.running = True

        engine_restart = self.engine.restart
        self.engine.restart = lambda: (order.append("engine"), engine_restart())
        self.svc.vision = FakeVision()
        self.assertEqual(self.svc.unload(), "unloaded")
        self.assertFalse(self.svc.vision.alive())
        self.assertEqual(self.chat()[0], 200)
        self.assertEqual(order, ["vision", "engine"])             # the encoder first, as at a start
        self.assertTrue(self.svc.vision.alive())

    def test_off_by_default(self):
        self.assertEqual((self.svc.idle_unload_s, self.svc.min_free_vram_mib, self.svc.before_load), (0, 0, None))
        self.assertEqual(self.req("/health")[1]["loaded"], True)

class HealthTellsACrashFromAnUnload(unittest.TestCase):
    """#56 review (xeno): after the 0.1.30 merge /health answered 200 for a crashed engine - its test for an unload
    (`not loaded()`) is also true after a crash.  A crash is 503 `engine_exited`; an unload on purpose (#208) is 200."""

    def health(self, crashed: bool):
        tok = ByteTokenizer()
        eng = UnloadableEngine(tok, "</think>\n\nok", max_context=CTX)
        eng.proc = ExitedProc()                         # the process has ended either way
        if crashed:
            eng.running = False
        else:
            eng.unload()
        svc = Service(eng, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{httpd.server_address[1]}/health", timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def test_a_crashed_engine_is_503(self):
        status, body = self.health(crashed=True)
        self.assertEqual((status, body["status"]), (503, "engine_exited"))

    def test_an_unloaded_engine_is_ok(self):
        status, body = self.health(crashed=False)
        self.assertEqual((status, body["status"], body["loaded"]), (200, "ok", False))

if __name__ == "__main__":
    unittest.main()
