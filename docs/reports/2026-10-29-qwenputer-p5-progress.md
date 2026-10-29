# Native Qwenputer P5: ownership and state foundations

Status: partial implementation. Decoder hooks and real-model
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

The implemented adaptation is a backward-compatible STQPUT01 minor 1 extension
with complete CPU-only token bytes and offsets, derived from the bound tokenizer.
All numeric circuit sections and the 1.0 runtime remain supported. The additional
raw data is 2,839,441 bytes, with no GPU upload. The data-only upgrade retains every
numeric section byte and records a new digest. Python container/lexicon and C++
loader/mutant controls pass 13 tests. Both artifact versions have equal derived
device-buffer budgets; the host estimate increases to include the new tables.
The first C++ RED attempt had a missing binding fixture and is not a valid RED;
the corrected fixture then failed at `artifact_version` before implementation.
The earlier v1 artifact and raw evidence remain unchanged. This extension
it satisfies the existing non-execution-of-quoted/history-text contract rather
than changing the approved execution architecture.

The new copyable `CallParser` performs no arithmetic. It restricts capture to
live enrolled assistant text, distinguishes other tool names, rejects malformed
calc framing/nesting/width/NUL/count errors, and suppresses entry in history,
quotes, inline/fenced/indented code, blockquotes and after an authoritative stop
token. It uses complete bytes, including the actual 32-tilde token. An initial
fence fixture incorrectly placed a close in the middle of a line; corrected tests
now check both a proper standalone close and a non-closing mid-line run. Added
boundary tests exposed indented-code entry and malformed multi-newline framing;
the same tests pass after their fixes. Decoder integration remains unimplemented.

## Tracker publication blocker

The user explicitly authorized phase/backlog updates. The existing issue contains
only its R4 body and normative appendix. A sanitized full English/Thai backfill was
prepared, but automatic approval review rejected the new comment before execution
because the earlier mandatory form was unresolved and this connector was treated
as a reroute. No alternate write or retry followed. The MCP Apps side panel exposed
no form; no approval ID or form URL is available. The user was asked to resolve or
cancel the stale form while implementation continues. No remote update is claimed.

## Final component verification for the lexicon checkpoint

The selected Python discovery run completed 99 tests: 95 passed and 4 skipped
in 30.879 seconds. The skips are the two unselected P0 probe tests and the
previously documented symbolic-link/junction environment controls. The native
CPU grammar test also passes using the actual bound tokenizer artifact, including
its complete 32-tilde token. CUDA rebuilt successfully; HIP/SYCL were not built
on this executor and remain explicit backend validation debt.

A prefix-only native CUDA smoke with artifact 1.1 computes `17*23` as `391`,
Success with 1,543 evaluations. It uses 31,804,988 owned device-buffer bytes,
equal to the previous standalone buffer count. Its artifact SHA256 is
`bc0db822ca05ee50a92b72adaeaf4a8c61d4a44b78846a091a2045c2b7b75152`;
the retained CLI SHA256 is
`d2503ac28469d1b97c82675fd01907b173216950b29d5a59f70cb777e46064c7`.
This is autonomous standalone circuit evidence, not model integration, an ABBA
measurement or proof of practical performance. The new CPU lexicon artifact and
CLI have separate identities from the frozen P4 held-out evidence.

Manual diff/boundary review checked loader offset overflow/ranges, versioned
section presence, unchanged numeric arrays and GPU uploads, bounded parser state,
copy isolation, role/stop gates and failure classification. The unavailable exact
review/security/simplify leaf skills were not claimed as run. No independent
whole-branch review or production decoder parity is claimed at this component
checkpoint. The standing final-branch review remains due after implementation.

The once-only comment retry after the user's explicit comment-vs-body
clarification was again rejected before execution by automatic approval review.
The stated blocker remains the unresolved earlier body-edit form. No remote
comment, alternative route or further retry followed; the backfill draft is safe.

## Transaction component checkpoint

`Transaction` now trials the candidate window on the actual circuit and a copied
CPU grammar state, restores its retained base, then replays only the accepted
input prefix. Its callback receives the clamped count and must commit and fence
the language verifier. The native burst runs only after that callback succeeds.
The controller never receives model caches. It stages all machine output until
Success and keeps an exact token-ID/cursor ledger so subsequent language replay
does not feed that text through the circuit again. A failed language callback,
restore, trace or mismatching replay invalidates the session. Lifecycle is serial;
the stream, executor and immutable artifact outlive it. The pinned input/status
arrays total 6,144 host bytes. History callers supply bounded chunks and check
cancellation between calls.

The first genuine RED failed at `transaction_not_implemented`; an earlier test
build failure was only a mixed-auto declaration in the fixture. Native GREEN
checks closing-call rows 0 through 4, discarded-shadow S/lag restoration,
rejected-tail decisions, exact callback/position counts and accepted-prefix
replay. A second control commits just the accepted newline before a rejected TCC,
then verifies that only the subsequent accepted TCC takes ownership. An owned
`17*23` burst returns `391`; replaying its result changes neither S/lag nor the
circuit Text-evaluation counter. These use a fake language callback, so actual
verifier, MTP, language-cache freeze and real-model parity remain open.

An allocation negative control reproduced a real defect: the controller rejected
an Executor/Artifact mismatch after allocating its base image. The corrected
initialization checks identities and explicit stream first, and runtime limits
before allocating the base image. Wrong identity, zero chunk limit and default
stream now fail with equal GPU-free-memory readings while the refused controller
is still alive. The complete functional test passes after the fix.

The bounded `--short` instrumentation control exercises ordinary Text steps,
trial/commit, rollback and injected language-commit failure. memcheck, initcheck
and synccheck report zero errors; racecheck reports zero hazards. This scope
does not launch the WHILE body or exercise result replay, and does not close the
P4 WHILE instrumentation gap. The new timeline analyzer fixture first failed
for missing transaction regions; all 22 timeline tests pass after implementation.
Host regions include waits and are not reported as invented GPU time.

No production decoder, service protocol, serving configuration or default has
changed at this checkpoint. HIP/SYCL validation remains unavailable, and the
whole P5 phase is still in progress.
