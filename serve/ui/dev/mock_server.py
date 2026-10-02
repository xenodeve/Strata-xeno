"""serve/ui/dev/mock_server.py - the web app's server for UI work: no model, no GPU, nothing the daily server owns.

    python serve/ui/dev/mock_server.py [port] [model.gguf ...]   (port default 8099)   then open http://127.0.0.1:8099/next/
    GGUF files given after the port are read for real (the model's name and quantization); without them a fixture shows.

What is fake: the answer text, and the engine's per-request STATS and prefill chunks (fixtures below, labelled as such).
STRATA_MOCK_MCP_CONFIG=file.json: the MCP servers of that run config are started and can be set up in Settings (issue #79).
STRATA_MOCK_AGENT=1: the chat's coding tools (issue #96) are on, and a message with "agent demo" in it makes the fake model use them step by step
(a todo list, a search, a read, a command that only reads, a file outside the folder, a command that asks): the server, the permission cards and the tools
are the real ones; only the model is a script.
STRATA_HOME=folder: the "other apps" whose skills and MCP servers Settings > Import shows are read from there, never from the real home (issue #94).
What is real: the hardware (GPUs, CPU, RAM, disks of THIS PC), the routes, the history on disk (a temp dir), the pages.
The daily server (:8091) is never touched; this one is its own process and port.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from serve import agent, gguf_info, harness, mcp_admin, shell  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.history import HistoryStore  # noqa: E402
from serve.server import IM_END, ByteTokenizer, MockEngine, Service, serve  # noqa: E402

THINKING = ("The user asks something open, so first I should work out what they actually want to know.\n\n"
            "There are two readings of the question. The narrow one has a short answer; the wide one needs the background first. I will check which one the wording supports, and note what each would cost to answer.\n\n"
            "The wide reading fits better: they mention the setup as well as the symptom. So the answer should start from the setup, then name the likely cause, then say how to confirm it.\n\n"
            "Before writing, check the numbers I intend to quote: nothing measured here, so say so instead of guessing. Keep the answer short, with one example and one list.")
ANSWER = (THINKING + "</think>\n\nสวัสดีครับ **bold** and `code`\n\n- one\n- two\n\n```py\nprint(1)\n```\n\n"
          "| a | b |\n|---|---|\n| 1 | 2 |\n")
if os.environ.get("STRATA_MOCK_LONG"):          # an answer longer than a screen, streamed slowly: for the chat's follow-the-answer scroll
    ANSWER = "\n\n".join([THINKING] * 2) + "</think>\n\n" + "\n\n".join(f"Paragraph {i}: " + "word " * 40 for i in range(1, 12))
THINK_BYTES = len(ANSWER.split("</think>")[0].encode())
THINK_MS = float(os.environ.get("STRATA_MOCK_THINK_MS", "30"))   # per token while it thinks (about 12 s of thinking by default): the thinking line stays on screen long enough to look at
PREFILL_TPS = float(os.environ.get("STRATA_MOCK_PREFILL_TPS", "300"))   # the prompt is read at this speed (the byte tokenizer: one token per byte, so paste a few thousand characters to see it)
PREFILL_CHUNK = 32
FAKE_STATS = dict(windows=40, tier_primary=5200, tier_secondary=1800, tier_pcie=300, tier_cpu=2700, cpu_expert_ms=950.5,
                  nvme_loads=12, nvme_ms=83.2, ms_verify=2100.0, ms_gpu_wait=800.0, ms_pool=1000.0, ms_plan=40.0,
                  ms_actq=60.0, ms_jobs=70.0, ms_cpu=800.0, ms_stage=120.0, ms_commit=60.0, ms_draft=90.0)


FIXTURE_MODEL = {"name": "Fixture Model 8B", "basename": "Fixture", "variant": "Q2_0", "source": "fixture", "size_label": "8B",
                 "architecture": "fixture", "files": ["fixture.gguf"], "bytes": 3_000_000_000, "bpw": 3.0,
                 "roles": [{"role": "experts", "tensors": 10, "bytes": 2_000_000_000, "types": ["Q2_0"], "bpw": 2.25},
                           {"role": "attention", "tensors": 20, "bytes": 1_000_000_000, "types": ["Q4_K", "BF16"], "bpw": 5.0}]}


class FakeEngine(MockEngine):
    """MockEngine with StrataEngine's clock (`last`), plus fixture STATS and prefill chunks."""

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        n = 0
        reused = min(5, len(ids))
        t0 = time.time()
        self.progress = (reused, len(ids))                   # as the engine's PP lines give it: the position counts the reused prefix
        pos = reused
        while pos < len(ids) and not cancel.is_set():        # the prompt is read at PREFILL_TPS tokens a second, in chunks
            step = min(PREFILL_CHUNK, len(ids) - pos)
            time.sleep(step / PREFILL_TPS)
            pos += step
            self.progress = (pos, len(ids))
        prompt_ms = (time.time() - t0) * 1000
        self.progress = None
        self.prefill_tok_s_mean = round((len(ids) - reused) / (prompt_ms / 1000), 1) if prompt_ms > 0 and len(ids) > reused else None
        try:
            for t in super().generate(ids, max_new, sampling, cancel, embeddings):
                n += 1
                time.sleep((THINK_MS if n <= THINK_BYTES else 4) / 1000)
                yield t
        finally:
            self.last = {"generated": n, "prompt_tokens": len(ids), "prompt_ms": prompt_ms, "decode_ms": 2400.0,
                         "finish": "stop", "reused": reused, "hits": 9, "lookups": 10,
                         "prefill_points": [(2005, 1000.0), (4005, 3000.0), (4805, 3400.0)], "stats": dict(FAKE_STATS)}


def _call(function: str, **params) -> str:
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"</think>\n\nLet me do the next step.\n\n<tool_call>\n<function={function}>\n{body}</function>\n</tool_call>"


class DemoAgentEngine(FakeEngine):
    """The fake model, but a message with "agent demo" in it is answered with a script of tool calls (one step per tool result so far)."""

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        text = bytes(t for t in ids if t < 256).decode("utf-8", "replace")
        at = text.rfind("agent demo")
        rw = text.rfind("rewind demo")                                  # a message with "rewind demo": a file is made and one that is there is changed, both inside the folder (free), so a rewind has something to put back
        if rw >= 0 and rw > at:
            step = text.count("<tool_response>", rw)
            script = [_call("Write", file_path="rewind-demo.txt", content="made by the scripted model\n"), _call("Read", file_path="existing.txt"),
                      _call("Edit", file_path="existing.txt", old_string="original text", new_string="changed by the model")]
            reply = script[step] if step < len(script) else "</think>\n\nDone: one file made and one changed."
            self.script = self.tok.encode(reply, parse_special=True) + self.tok.encode(IM_END, parse_special=True)
            yield from super().generate(ids, max_new, sampling, cancel, embeddings)
            return
        if at >= 0:
            step = text.count("<tool_response>", at)
            outside = os.path.join(tempfile.gettempdir(), "strata-agent-demo.txt").replace("\\", "/")
            if step == 0 and os.path.exists(outside):                  # a demo that is run again starts from the same place (a file that is there must be read before it is written)
                os.remove(outside)
            todos = json.dumps([{"content": "Look around", "status": "in_progress", "activeForm": "Looking around"}, {"content": "Make two files", "status": "pending", "activeForm": "Making two files"}])
            script = [_call("TodoWrite", todos=todos), _call("Glob", pattern="*"), _call("Read", file_path="README.md", limit=5), _call("Bash", command="git status --short", description="What changed"),
                      _call("Write", file_path=outside, content="written by the scripted model\n"), _call("Bash", command="touch strata-agent-demo.txt", description="Make a file here")]
            reply = script[step] if step < len(script) else "</think>\n\nThat was the demo of the coding tools: a list of steps, a search, a read, a command that only reads, a file outside the folder and a command that asked. Nothing else was changed."
            end = self.tok.encode(IM_END, parse_special=True)
            self.script = self.tok.encode(reply, parse_special=True) + end
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


def main(port: int = 8099, model_files: list[str] | None = None):
    tok = ByteTokenizer()
    demo = bool(os.environ.get("STRATA_MOCK_AGENT"))
    svc = Service((DemoAgentEngine if demo else FakeEngine)(tok, ANSWER, max_context=400_000 if demo else 4096), tok, ChatTemplate(ROOT / "serve" / "chat_template.jinja"))
    if demo:                                                         # the coding tools: the real ones, with a fake model
        svc.agent = agent.AgentServer()
        shell.install(svc.agent, shell.find_shell())
    svc.gpu_indices = [0, 1]                                         # as monitor_gpus() lists them on the real server
    svc.engine.info = {"cpu_isa": "AVX-VNNI", "pool_workers": 13, "kv": "int8", "engine": "0.1.30 (fixture)", "mtp_max": 4, "lookup": 3, "spec": 6,    # fixture: what a real engine's INFO lines carry
                       "gpu_arch": "sm120@NVIDIA_GeForce_RTX_5060_Ti,sm89@NVIDIA_GeForce_RTX_4070_SUPER"}
    svc.model_info = gguf_info.model_info(model_files or []) or FIXTURE_MODEL     # real headers when files are given
    svc.hstore = HistoryStore(tempfile.mkdtemp(prefix="strata-ui-history-"))
    config = os.environ.get("STRATA_MOCK_MCP_CONFIG")
    if os.environ.get("STRATA_HOME"):                # the skills and MCP servers of "other apps" are read from this (fake) folder: Settings > Import (#94)
        svc.importer = harness.Importer()
        svc.importer.rescan(json.loads(Path(config).read_text(encoding="utf-8")) if config else {})
    if config:                                       # a run config file (JSON with "mcp_servers"): the MCP servers can be set up in Settings
        svc.config_path = config
    if config or svc.importer is not None or svc.agent is not None:
        mcp_admin.reload_hub(svc)
    httpd = serve(svc, port=port)
    print(f"mock server on http://127.0.0.1:{port}/  (new app at /next/, classic at /classic/)", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8099, sys.argv[2:])
