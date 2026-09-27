# Claude Code compatibility slices

Engine issue: xenodeve/Strata-xeno#6. Measurement issue: xenodeve/Qwen3.8-Flash-Next-Tuning#15.

## Anthropic count_tokens

The server now exposes `POST /v1/messages/count_tokens` through the same Anthropic message conversion, chat template and tokenizer used by generation. It reports the full rendered prompt token count even when that count exceeds the model context, so the client can decide when to compact. For images it uses the existing vision encoder's expansion count and removes the per-request combined embedding file after returning.

The endpoint was red as HTTP 404 before the change. `test_count_tokens.py` now passes an over-context text request, an image request that expands one placeholder to three tokens, and a failed image-source read after the request file has been created. The latter test was red because the partial file remained; the server now removes it, and normal count requests remove staging before sending the response. No GPU/model run was required for this slice; the image encoder is represented by a three-token fake at the HTTP seam.

The remaining Anthropic behavior, guards, PDF handling, watchdog, hub profile and rollback are still open. PR #10's short-read mode remains parked for token divergence and is not a prerequisite for count_tokens.

## Claude Code billing header

The Anthropic converter strips the leading observed `cc_*` and `cch` billing fields from system text before rendering. Two requests whose only difference is the per-request `cch` stamp now produce identical rendered token IDs. The red/green test covers string and text-block system forms, while user text and assignment-shaped system instructions after the header are preserved. Unrecognized billing field names remain in the prompt until their format is established. This is prompt construction behavior; a long-session prefix-cache timing test remains for the final serving gate.

## Anthropic stream and loop guard

The Anthropic stream now emits an empty `signature_delta` before closing a thinking content block. A focused event test was red before the change and green after; the same test confirms streamed tool JSON arrives once before `tool_use` stop. `LoopGuard` was adapted from the existing EXL3 server and is fed only reasoning/content text in `Service.run`. It cancels a degenerate request after a 512-character single-character loop or a 64-character Thai-script micro-loop, increments `loops_stopped` on `/health`, and leaves the tested normal prose running. The mock-engine loop test was red at 800 repeated tokens before the change and green at 512 after. A live model runaway was not induced for validation.

The internal finish detail and OpenAI streamed/non-streamed `timings.stop_reason` report `loop`. Anthropic responses retain the standard `max_tokens` stop reason, matching the EXL3 compatibility layer, because [`loop` is not an Anthropic stop_reason value](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons). `/status.last_stop_reason` updates at every request completion, and the server log exposes the exact cause; Claude Code does not receive that custom detail in its Anthropic response.
