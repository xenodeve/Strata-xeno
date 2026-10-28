# P0 diagnostic review — checkpoint c6e01117

Fixed point: e5737a27f485773a7ce22784fef6af7b480dcc9a. Diff: `git diff e5737a27...c6e01117`.
Scope: P0 diagnostics plus the explicitly authorized startup investigation; P1+ remains outside scope.

## Standards

Hard documented-standard violations: none found in the P0 diff.

Judgment call — Low, possible Duplicated Code: headless.py:20–28 and memory.py:30–39 independently define the same Windows basic/extended job-limit ABI layout. Both layouts govern the same job object, so a later ABI correction could update one while leaving the other stale. Consider sharing the structures. This is a smell-baseline heuristic, not a documented-rule breach.

The diagnostic remains opt-in; the blueprint records its responsibilities and cleanup mechanism. All inspected console-process launch sites pass CREATE_NO_WINDOW. The explicit diagnostic cap removal is authorized. No review blocker.

Read-only inspection; no tests, builds, model execution, edits or publication by the Standards reviewer.

## Spec

No findings in the completed P0/startup-diagnosis scope. Inventory, isolated execution and conditional-graph diagnostics implement the P0 deliverables. Explicit None changes only the owned process cap and preserves unrelated flags. Recorded run05 confirms null process/job caps, generation, secondary compute and MTP activity. The over-budget baseline is reported and resource compliance remains unresolved.

Candidate raw-token parity, ABBA performance, backend validation and resource/release gates remain open and are disclosed. P1+ circuit/bridge work is neither implemented nor claimed. No scope creep identified.

Read-only inspection; no tests, builds, models, edits or publication by the Spec reviewer.

## Scrutinize — owner end-to-end pass

Goal: distinguish a test guard from OS commitment failure, then restore the selected baseline workload without a production behavior change.

A one-line cap removal could let the test proceed, but would not satisfy the requested attribution. The separate system/process/job telemetry, positive small limit control and failure-injection tests serve that requirement. No engine arithmetic or allocator rewrite is justified by these results.

Trace: no-cap controller → OwnedJob kill-on-close → JobMemoryMonitor explicit None → QueryInformationJobObject readback → assignment before worker handshake → unchanged baseline → full host-expert load → prefill → verifier/MTP/secondary execution → 16 output IDs → job drain → durable report. Successful run05 peaks above42GB while the actual job reports no hard process/job cap. This refutes a hidden equivalent42GB supervisor threshold in the tested path.

Error paths: queued events survive later API errors; failed enrichment is retained; optional post-run probes cannot erase the initial report; cleanup is nested. Tests exercise these boundaries. No Critical/Important finding remains. The Low ABI-duplication suggestion is deferred: sharing ctypes structures would expand the proven launcher change for a maintenance-only benefit, with no startup fix needed now.

Verdict: the local diagnostic correction is verified. This is not Native feature completion, a release approval, a budget-policy change or PRD publication.

## Per-item gates

| Item | simplify | verify | code-review | scrutinize | security-review |
|---|---|---|---|---|---|
| P0 memory diagnosis/cap exception | not-run: named skill absent in checked local roots; manual simpler-alternative pass above | ran: 23 tests,21pass,2GPUskip and one real same-input inference run | ran: independent Standards/Spec axes against fixed point | ran: owner trace and error-path review above | n-a: task-owned diagnostic module; no auth/security module or global permissions changed |
| PRD R6 status preparation | n-a: prose only | not-run: GitHub readback of R6 blocked by pending R5 Submit; privacy and unchanged Mermaid-source checks passed locally | n-a: no runtime code in this item | ran: claims checked against run05 and preserved contracts | n-a: no private paths/IDs in public draft |
| Task Manager image inspection | n-a: bundled helper unchanged | not-run: no readable PNG remains after Windows xattr failure | n-a | not-run: actual pixels unavailable | n-a: no sharing/publication |

No push, PR, merge, deployment or unrelated tracker item was authorized or performed. The existing GitHub Submit form is parked; it was not retried or bypassed.

## Actual-hub follow-up review

Spec axis: no material findings in the actual-hub controller, result and closeout. The daily executable and unchanged-main control remain distinct. Sixteen reasoning tokens do not establish a final answer, parity or performance.

Standards axis found one Important issue in the first hub controller: the unchanged deployed server launches console children without CREATE_NO_WINDOW. Outer Python flags alone did not satisfy the repository rule. Run06's no-window claim was withdrawn. A task-only server_no_window_shim.py now wraps subprocess.Popen before runpy loads the unchanged server, ORs CREATE_NO_WINDOW and preserves other flags. Static reproduction plus a real small child control returned GetConsoleWindow=false. The reviewer found no remaining hard violation in this correction.

Run07 through that shim completed with an API response containing 16 reasoning tokens, null hard/notification limits, no emergency stop, no telemetry errors and verified owned-job drain. The source server, served config and original learned profile remain unchanged. This is a corrected diagnostic launcher, not a production engine or serving-default change. The operational-baseline contract is prepared in local PRD R7; public synchronization remains pending and is not claimed.
