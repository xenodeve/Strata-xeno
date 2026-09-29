"""serve/test_server.py - the max tokens budget over both APIs, against the mock engine (no GPU, no pack).

    python -m unittest serve.test_server -v
"""
from __future__ import annotations

import io
import json
import os
import queue
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import CTX_SLACK, ByteTokenizer, EngineDied, MockEngine, Service, StrataEngine, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CTX = 4096
# longer than the old 1024 fallback, one token per byte, and not a loop: "x" * 2000 is one, and the serving loop
# guard (serve/loop_guard.py, xeno) rightly stops it at 512 characters
ANSWER = "".join("abcdefghijklmnopqrstuvwxyz"[(i * 7919 + i * i * 104729) % 26] for i in range(2000))


class RecordingEngine(MockEngine):
    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.last_max_new = max_new
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
        self.assertIn(b"/web/app.js", body)
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


if __name__ == "__main__":
    unittest.main()
