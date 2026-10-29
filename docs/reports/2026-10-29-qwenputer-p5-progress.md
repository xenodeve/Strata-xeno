# Native Qwenputer P5: ownership and state foundations

Status: partial implementation. Decoder hooks, native-call grammar and real-model
ownership/freeze/replay gates are not implemented yet. P4 functional checkpoint
is `10d848eb`; its WHILE instrumentation coverage remains open.

## Completed component checks

The CPU ownership helper distinguishes consumed INPUT rows from emitted NEXT
picks. An accepted takeover at row r commits r+1 inputs and exposes r language-model
picks, suppressing the language bonus at the closing-call row. A candidate in the
rejected suffix changes nothing. All window lengths 1 through 32, accepted lengths and
takeover positions are covered, together with transactional invalid arguments.
The stub produced 53,328 assertion failures before implementation; the completed
helper passes the same native test.

GPU StateImage stores S, lag history, input and prediction status only. It owns
13,195,240 bytes for this artifact, does not copy model caches or CUDA graph
handles, and requires one explicit created ordered stream. Tests mutate the real
circuit state, roll it back byte for byte, restore into an Executor with a distinct
GPU allocation, and obtain identical continuation state. Negative controls refuse
uninitialized/reinitialized images and mismatched artifact/device/stream before
copying. An initial negative test caught implicit default-stream acceptance;
the guard now refuses default aliases explicitly. Callers must fence completion
and invalidate a session on restore failure; an immediate capture-copy failure leaves the image invalid. Completion still
requires the caller’s fence.

The regular state-copy test passes memcheck, initcheck, synccheck and racecheck
using the installed sanitizer's named-pipes connection. These results do not
close the independent P4 WHILE instrumentation gap.

## Next implementation dependency

The circuit's 16-byte token projections are sufficient for its bounded capture,
but are insufficient for a correct outer quote/fence/history lexer. The actual
selected tokenizer has 6,882 tokens longer than 16 bytes; 38 of those contain a
newline, and token 63434 contains 32 tildes. Inspecting only a projection could
miscount a fence or miss a line boundary.

The next planned adaptation is a backward-compatible STQPUT01 minor 1 extension
with complete CPU-only token bytes and offsets, derived from the bound tokenizer.
All numeric circuit sections and the 1.0 runtime remain supported. The additional
raw data is about 2.84 MB, with no GPU upload. This extension is not implemented yet;
it satisfies the existing non-execution-of-quoted/history-text contract rather
than changing the approved execution architecture.

## Tracker publication blocker

The user explicitly authorized phase/backlog updates. The existing issue contains
only its R4 body and normative appendix. A sanitized full English/Thai backfill was
prepared, but automatic approval review rejected the new comment before execution
because the earlier mandatory form was unresolved and this connector was treated
as a reroute. No alternate write or retry followed. The MCP Apps side panel exposed
no form; no approval ID or form URL is available. The user was asked to resolve or
cancel the stale form while implementation continues. No remote update is claimed.
