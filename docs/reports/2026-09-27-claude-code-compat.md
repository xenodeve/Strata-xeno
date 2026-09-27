# Claude Code compatibility slices

Engine issue: xenodeve/Strata-xeno#6. Measurement issue: xenodeve/Qwen3.8-Flash-Next-Tuning#15.

## Anthropic count_tokens

The server now exposes `POST /v1/messages/count_tokens` through the same Anthropic message conversion, chat template and tokenizer used by generation. It reports the full rendered prompt token count even when that count exceeds the model context, so the client can decide when to compact. For images it uses the existing vision encoder's expansion count and removes the per-request combined embedding file after returning.

The endpoint was red as HTTP 404 before the change. `test_count_tokens.py` now passes an over-context text request, an image request that expands one placeholder to three tokens, and a failed image-source read after the request file has been created. The latter test was red because the partial file remained; the server now removes it, and normal count requests remove staging before sending the response. No GPU/model run was required for this slice; the image encoder is represented by a three-token fake at the HTTP seam.

The remaining Anthropic behavior, guards, PDF handling, watchdog, hub profile and rollback are still open. PR #10's short-read mode remains parked for token divergence and is not a prerequisite for count_tokens.
