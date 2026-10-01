"""serve/ui/dev/mock_server.py - the web app's server for UI work: no model, no GPU, nothing the daily server owns.

    python serve/ui/dev/mock_server.py [port] [model.gguf ...]   (port default 8099)   then open http://127.0.0.1:8099/next/
    GGUF files given after the port are read for real (the model's name and quantization); without them a fixture shows.

What is fake: the answer text, and the engine's per-request STATS and prefill chunks (fixtures below, labelled as such).
What is real: the hardware (GPUs, CPU, RAM, disks of THIS PC), the routes, the history on disk (a temp dir), the pages.
The daily server (:8091) is never touched; this one is its own process and port.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from serve import gguf_info  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.history import HistoryStore  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ANSWER = ("Let me think about this.</think>\n\nสวัสดีครับ **bold** and `code`\n\n- one\n- two\n\n```py\nprint(1)\n```\n\n"
          "| a | b |\n|---|---|\n| 1 | 2 |\n")
if os.environ.get("STRATA_MOCK_LONG"):          # an answer longer than a screen, streamed slowly: for the chat's follow-the-answer scroll
    ANSWER = "Thinking it through. " * 24 + "</think>\n\n" + "\n\n".join(f"Paragraph {i}: " + "word " * 40 for i in range(1, 15))
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
        try:
            for t in super().generate(ids, max_new, sampling, cancel, embeddings):
                n += 1
                time.sleep(0.004)
                yield t
        finally:
            self.last = {"generated": n, "prompt_tokens": len(ids), "prompt_ms": 40.0, "decode_ms": 2400.0,
                         "finish": "stop", "reused": min(5, len(ids)), "hits": 9, "lookups": 10,
                         "prefill_points": [(2005, 1000.0), (4005, 3000.0), (4805, 3400.0)], "stats": dict(FAKE_STATS)}


def main(port: int = 8099, model_files: list[str] | None = None):
    tok = ByteTokenizer()
    svc = Service(FakeEngine(tok, ANSWER, max_context=4096), tok, ChatTemplate(ROOT / "serve" / "chat_template.jinja"))
    svc.gpu_indices = [0, 1]                                         # as monitor_gpus() lists them on the real server
    svc.engine.info = {"cpu_isa": "AVX-VNNI", "pool_workers": 13, "kv": "int8", "engine": "0.1.30 (fixture)", "mtp_max": 4, "lookup": 3, "spec": 6,    # fixture: what a real engine's INFO lines carry
                       "gpu_arch": "sm120@NVIDIA_GeForce_RTX_5060_Ti,sm89@NVIDIA_GeForce_RTX_4070_SUPER"}
    svc.model_info = gguf_info.model_info(model_files or []) or FIXTURE_MODEL     # real headers when files are given
    svc.hstore = HistoryStore(tempfile.mkdtemp(prefix="strata-ui-history-"))
    httpd = serve(svc, port=port)
    print(f"mock server on http://127.0.0.1:{port}/  (new app at /next/, classic at /classic/)", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8099, sys.argv[2:])
