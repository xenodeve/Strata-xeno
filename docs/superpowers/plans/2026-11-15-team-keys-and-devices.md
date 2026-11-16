# Team keys, roles and device approval - Implementation Plan (part 1 of Strata Desktop)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give each team member a named key with a role (owner / guest), hold any key used from a new device until it is
approved, scope runs, prompts and Monitor content to their identity, and let the owner manage keys live from Settings.

**Architecture:** A new module `serve/identity.py` owns the keys file (named, hashed keys with their devices) and the
decisions (who is this request, may it act, may it see this record). `server.py`'s handler resolves one `Identity` per
request in `_authorized` and every gate asks it instead of "is a key set". The web app sends a device id with every
request, shows a waiting screen while its device is pending, and gains a "Keys and devices" settings section and an
approval notice.

**Tech Stack:** Python 3.11 standard library (`http.server`, `hashlib`, `hmac`, `secrets`, `json`), unittest;
React + TypeScript web app in `serve/ui` (bun test, vite build, `dist/` committed).

**Spec:** `docs/superpowers/specs/2026-11-15-strata-desktop-design.md` (sections "Identity and roles", "Decisions by
the developer after the builder check" 1-3, "Calls made by the designer", anti-goals 1, 2 and 5).

Parts 2 (runner and the two tool groups) and 3 (the Strata Desktop program) get their own plans after this one lands.

## Global Constraints

- Default path byte-identical: a server with no keys file and a single `"api_key"` (or none) behaves exactly as today,
  except that the routes this plan makes owner-only stay open to that single key (it is the owner).
- Standard library only on the server side.
- Roles are exactly `"owner"` and `"guest"`. The legacy `"api_key"` / `--api-key` / `STRATA_API_KEY` keys are
  identity `{"name": "owner", "role": "owner"}`.
- Keys file: config key `"keys_file"`, default `strata-keys.json` in the run config's folder; no run config -> no keys
  file. New keys are `"sk-strata-" + secrets.token_urlsafe(24)`, stored only as `sha256` hex, shown once at creation.
- Device header: `X-Strata-Device` (id, 8-64 chars of `[A-Za-z0-9_-]`) and `X-Strata-Device-Name` (display, <= 80
  chars). A request without it is the device `addr:<ip>`.
- Implicit approval only for a request whose socket peer is loopback **and** whose Host is a loopback name
  (`mcp_admin.host_is_loopback`) **and** that carries none of `Cf-Connecting-Ip`, `X-Forwarded-For`, `Forwarded`.
  Forwarding headers name the device address only when the socket peer is loopback.
- Pending answer: HTTP 403 `{"error": {"type": "device_pending", "code": "<4 chars A-Z0-9>", "message": ...}}`;
  refused: 403 `{"error": {"type": "device_refused", ...}}`. `/health`, `/` and static files never need a device.
- `docs/BLUEPRINT.md` is updated in the same commit as each task that adds a route, config key, header or file
  (`tools/hooks/commit-msg` enforces part of it). Issue comments in English + Thai on `xenodeve/Strata-xeno` #310.
- Python tests: `python -m pytest serve/<file> -q` from the repo root. UI: `cd serve/ui && bun test`, then
  `bun run build` and commit `dist/` (enforced by `serve/test_ui.py`).

## Review Focus

1. **A Cloudflare Tunnel or `tailscale serve` request arrives from 127.0.0.1.** It must not be implicitly approved: its
   Host is the public name and it carries forwarding headers. Test in Task 2 (`test_tunnel_request_is_not_loopback`).
2. **A forged `X-Forwarded-For` from a remote peer** must not choose its device address. Test in Task 2
   (`test_forwarded_for_from_remote_peer_is_ignored`).
3. **A key revoked mid-run.** Its running runs are cancelled and its next request is 401. Test in Task 5
   (`test_revoke_cancels_runs_and_refuses_next_request`).
4. **The keys file is edited or broken by hand while the server runs.** A file that fails to parse keeps the last good
   keys and logs one line; it never opens the server. Test in Task 1 (`test_broken_file_keeps_last_good_keys`).
5. **Claude Code on an engineer's PC** (no device header) shows up as a pending `addr:<ip>` device with its client
   name, and the owner can approve it. Test in Task 2 (`test_headerless_client_is_pending_by_address`).

---

### Task 1: `serve/identity.py` - the keys file and its decisions

**Files:**
- Create: `serve/identity.py`
- Test: `serve/test_identity.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class Identity: name: str; role: str  # "owner" | "guest"`; `Identity.is_owner -> bool`.
  - `OWNER = Identity("owner", "owner")`, `LOCAL_GUEST = Identity("anonymous", "guest")`.
  - `class KeyStore(path: str | None, legacy: str | list[str] = "")`
    - `.match(given: str) -> Identity | None` (constant time over every key, as `server.key_matches`; legacy keys ->
      `OWNER`).
    - `.any_key() -> bool`.
    - `.reload_if_changed() -> None` (mtime check; called by `match`, at most once per second).
    - `.create(name: str, role: str) -> str` (returns the plain key once), `.rename(name, new)`, `.set_role(name, role)`,
      `.revoke(name) -> bool`, `.view() -> list[dict]` (`name`, `role`, `created`, `devices`; never a key or hash).
    - `.device(ident: Identity, device_id: str, label: str, addr: str, now: float) -> str` returns `"approved"`,
      `"pending"` or `"refused"`, recording a new device as pending with a 4-char code.
    - `.decide_device(key_name: str, device_id: str, approve: bool, by: Identity) -> bool`;
      `.clear_device(key_name, device_id) -> bool`; `.pending() -> list[dict]` (`key`, `device`, `label`, `addr`,
      `code`, `first_seen`).
    - `.may_approve(by: Identity, key_name: str, by_device: str | None) -> bool`: owner always; else only for its own
      key and only from an approved device of it; never for a key with no approved device yet.
  - `visible_record(record: dict, viewer: Identity) -> dict`: the record unchanged for the owner or when
    `record.get("who") == viewer.name`; otherwise without `preview`, `input`, `output`, `reasoning`, `response`,
    `prompt`, `messages` and any key holding a detail.
- File format (JSON, written atomically, private ACL like `history._private_dir`): `{"keys": [{"name", "role",
  "sha256", "created", "devices": {"<id>": {"label", "addr", "status", "code", "first_seen", "last_seen",
  "decided_by"}}}], "log": [{"t", "what", "key", "device", "by"}]}`; `log` capped at 500 entries.

- [ ] **Step 1: Write the failing tests** in `serve/test_identity.py` (unittest, temp dir):
  - `test_legacy_key_is_owner`: `KeyStore(None, "abc").match("abc") == OWNER`; `match("x") is None`.
  - `test_created_key_matches_and_is_stored_hashed`: `k = s.create("ann", "guest")`; `k.startswith("sk-strata-")`;
    `s.match(k) == Identity("ann", "guest")`; the file text does not contain `k`.
  - `test_names_are_unique_and_roles_checked`: second `create("ann", ...)` raises `ValueError`;
    `create("bob", "admin")` raises `ValueError`.
  - `test_revoke_removes_the_key`: after `revoke("ann")`, `match(k) is None`.
  - `test_new_device_is_pending_then_approved`: `device(ann, "d1", "PC-1", "26.1.1.1", t)` -> `"pending"`; one
    `pending()` entry with a 4-char code; `decide_device("ann", "d1", True, OWNER)`; then `device(...)` ->
    `"approved"`.
  - `test_refused_device_stays_refused_until_cleared`.
  - `test_may_approve_rules`: owner yes; ann from approved d1 for key ann yes; ann for key bob no; ann from pending d2
    no; ann for her own key with no approved device no.
  - `test_reload_after_external_edit`: write a new key into the file by hand with a later mtime -> `match` finds it.
  - `test_broken_file_keeps_last_good_keys` (Review Focus 4): overwrite with `"{"` -> previous keys still match, and
    `match` of a random string is still `None`.
  - `test_visible_record`: owner sees `preview`; `Identity("ann","guest")` sees her own record's `preview` and not
    bob's, but sees bob's `state`, `who`, `queue_s`.

- [ ] **Step 2: Run** `python -m pytest serve/test_identity.py -q` - Expected: FAIL (`ModuleNotFoundError: serve.identity`).
- [ ] **Step 3: Implement** `serve/identity.py` as specified above; one `threading.Lock` around file state.
- [ ] **Step 4: Run** `python -m pytest serve/test_identity.py -q` - Expected: all pass.
- [ ] **Step 5: Commit** `feat(serve): named keys with roles and devices (serve/identity.py, #310)`; BLUEPRINT gets
  the keys file in its files/config section (`"keys_file"`).

### Task 2: Resolve the identity and the device on every request

**Files:**
- Modify: `serve/server.py` (`Service.__init__` near 3021: add `self.keys: identity.KeyStore`, `self.trust_loopback =
  True`; `_authorized` 5539; `parse_request` / GET and POST entry points; startup near 7614 building the KeyStore from
  `"keys_file"` and the legacy key)
- Modify: `serve/mcp_admin.py:51-60` (`may_edit` keeps its signature for its tests; server calls stop using it)
- Test: `serve/test_identity_http.py` (start helper as in `serve/test_security.py:99-136`)

**Interfaces:**
- Consumes: Task 1 `KeyStore`, `Identity`, `OWNER`, `LOCAL_GUEST`.
- Produces (on the handler):
  - `self.identity: Identity | None` set by `_authorized() -> bool`. No key configured at all: `OWNER` for an implicit
    loopback request, `LOCAL_GUEST` otherwise (today's `may_edit` semantics).
  - `self.device_id: str`, `self._loopback_local() -> bool` (the Global Constraints rule; `svc.trust_loopback = False`
    in tests turns it off).
  - `self._owner_only(what: str) -> bool`: answers 403 `{"error": {"type": "forbidden", "message": f"{what} only with
    an owner key"}}` and returns False for a non-owner.
  - Device gate inside `_authorized` after a key matched: `svc.keys.device(...)`; pending/refused -> the 403s from
    Global Constraints, return False. Implicit loopback requests skip it.

- [ ] **Step 1: Write the failing tests:**
  - `test_single_legacy_key_is_owner_everywhere`: `api_key="k"`, a non-loopback request (`trust_loopback=False`) with
    `X-Strata-Device: d1` is first `device_pending`; after `svc.keys.decide_device("owner", "d1", True, OWNER)`,
    `GET /mcp/config` -> `editable: true`.
  - `test_guest_cannot_edit`: guest key with an approved device: `POST /mcp/config` -> 403 `forbidden`;
    `GET /agent` -> `allowed: false`; `POST /v1/chat/completions` -> 200.
  - `test_owner_only_routes`: for a guest, each of `POST /settings`, `/config`, `/import`, `/unload`, `/load`,
    `/v1/vram`, `/slots/0?action=save`, `/metrics/keep`, `/agent/hooks`, `/agent/web`, `/agent/helpers` -> 403
    `forbidden`; `GET /agent/folders`, `/agent/git`, `/agent/files`, `/agent/memory` -> 403.
  - `test_pending_device_gets_code_and_health_stays_open`: pending -> 403 `device_pending` with a 4-char `code`;
    `GET /health` -> 200; `GET /` -> 200.
  - `test_tunnel_request_is_not_loopback` (RF 1): peer 127.0.0.1, `Host: strata.example.com`, `Cf-Connecting-Ip:
    203.0.113.5`, owner key, new device -> `device_pending`.
  - `test_forwarded_for_from_remote_peer_is_ignored` (RF 2): with `trust_loopback=False` and the peer treated as
    remote, `X-Forwarded-For: 26.9.9.9` does not change the recorded `addr`.
  - `test_headerless_client_is_pending_by_address` (RF 5): guest key, no device header, `User-Agent: claude-cli/2` ->
    pending entry with device `addr:<ip>` and label containing `claude-cli`.
  - `test_no_key_configured_is_unchanged`: no keys at all: loopback `POST /mcp/config` works as before; the existing
    `serve/test_security.py` and `serve/test_mcp_admin.py` still pass.
- [ ] **Step 2: Run** `python -m pytest serve/test_identity_http.py -q` - Expected: FAIL.
- [ ] **Step 3: Implement.** Replace every `mcp_admin.may_edit(bool(svc.api_key), ...)` call (list in the spec's
  appendix) with `self.identity.is_owner` / `self._owner_only(...)`; the "allowed"/"editable" fields in GET answers
  come from `self.identity.is_owner`. Add `_owner_only` to the POST routes listed in `test_owner_only_routes`.
  `strata_agent` in a chat request (6493) is owner-only; `strata_mcp` and `strata_skill` stay open.
- [ ] **Step 4: Run** `python -m pytest serve/test_identity_http.py serve/test_security.py serve/test_mcp_admin.py serve/test_server.py -q` - Expected: all pass.
- [ ] **Step 5: Commit** `feat(serve): one identity per request, owner-only gates, device approval (#310)`; BLUEPRINT:
  the two headers, the 403 types, the owner-only route list.

### Task 3: Runs and prompts belong to their identity

**Files:**
- Modify: `serve/runs.py` (`Run.__init__` 24, `RunStore.launch` 86, `cancel` 122, `steer` 129, new `get_for`)
- Modify: `serve/agent_run.py` (`Broker.open` 25, `answer` 35, `respond` 47; `AgentRun` passes its identity name)
- Modify: `serve/server.py` (`/agent/run` GET, `/agent/steer`, `/agent/cancel`, `/agent/permission`,
  `/agent/question`, `/agent/rewind`, `/agent/checkpoints/forget`, run launch)
- Test: `serve/test_runs.py`, `serve/test_identity_http.py`

**Interfaces:**
- Produces: `Run.who: str`; `RunStore.launch(..., who: str = "owner")`; `RunStore.get_for(rid, viewer: Identity) ->
  Run | None` (the owner sees all, others only `who == viewer.name`); `RunStore.cancel(rid, by: Identity) -> bool`
  (owner any run; others their own); `RunStore.steer(rid, text, by: Identity) -> bool`;
  `RunStore.cancel_all(who: str) -> int`; `Broker.open(rid, kind="permission", who="owner")`;
  `Broker.answer(rid, decision, by: Identity)` / `respond(rid, answers, by: Identity)` (only `slot["who"] == by.name`;
  the owner may not answer for others - an approval is the run holder's).
- Checkpoints (`/agent/rewind`, `/agent/checkpoints/forget`) key on the chat session id; they become owner-only in
  this part (guests have no server folders); part 2 moves My-PC checkpoints to the runner.

- [ ] **Step 1: Write the failing tests:** in `serve/test_runs.py` `test_get_for_scopes_by_identity`,
  `test_cancel_by_owner_or_holder_only`, `test_steer_only_by_holder`, `test_cancel_all_by_who`; in
  `serve/test_identity_http.py` `test_guest_cannot_read_or_answer_another_guest_run`: guest ann launches a run
  (MockEngine scripted tool call that asks permission); guest bob's `GET /agent/run?id=` -> 404 and his `POST
  /agent/permission` -> 404; ann's answer -> 200.
- [ ] **Step 2: Run** `python -m pytest serve/test_runs.py serve/test_identity_http.py -q` - Expected: FAIL.
- [ ] **Step 3: Implement** the signatures above; routes answer 404 (not 403) for another identity's id so ids are not
  probed.
- [ ] **Step 4: Run** `python -m pytest serve/test_runs.py serve/test_agent_run.py serve/test_identity_http.py serve/test_server.py -q` - Expected: all pass.
- [ ] **Step 5: Commit** `fix(serve): runs, permission and question prompts belong to their identity (#310)`.

### Task 4: Monitor, history and the queue by identity

**Files:**
- Modify: `serve/server.py` (`begin_request` 3693: `record["who"]`; `request_records`; `metrics` 3724; the history
  record written per request; `/metrics`, `/metrics/requests`, `/metrics/requests/<id>`, `/api/requests` routes; the
  place where `status["queued"]` is incremented and decremented)
- Modify: `serve/history.py` (`summary_record`, `page`, `summary`, `detail` keep `who`)
- Test: `serve/test_identity_http.py`, `serve/test_monitor.py`, `serve/test_history.py`

**Interfaces:**
- Consumes: Task 1 `visible_record`, Task 2 `self.identity`.
- Produces: every record and history summary has `"who": str`; `Service.metrics(all_requests=False, viewer:
  Identity = OWNER)`; `live["holder"]: str | None` (identity holding the engine, or each slot's `who` in
  `live["slots"]` with `"parallel"`); `live["waiting"]: list[{"who": str, "since": float}]` in arrival order; every
  record list passes through `visible_record(r, viewer)`.

- [ ] **Step 1: Write the failing tests:** `test_guest_sees_others_without_content` (ann and bob each make a request;
  bob's `GET /metrics/requests` shows ann's record with `who`, `state`, timings and no `preview`; bob's own has
  `preview`; owner sees both); `test_detail_of_another_guest_is_404`; `test_live_shows_holder_and_waiting` (MockEngine
  with `delay_s`, two concurrent requests from ann and bob: `live.holder == "ann"`, `live.waiting[0].who == "bob"`);
  `test_history_keeps_who` in `serve/test_history.py`.
- [ ] **Step 2: Run** `python -m pytest serve/test_identity_http.py serve/test_monitor.py serve/test_history.py -q` - Expected: FAIL.
- [ ] **Step 3: Implement** as above. `POST /metrics/keep` is already owner-only (Task 2).
- [ ] **Step 4: Run** the same command plus `serve/test_server.py` - Expected: all pass.
- [ ] **Step 5: Commit** `feat(serve): who holds the engine and who waits; Monitor content by identity (#310)`.

### Task 5: Key and device management routes

**Files:**
- Modify: `serve/server.py` (GET/POST routes), Test: `serve/test_identity_http.py`

**Interfaces:**
- Consumes: Task 1 KeyStore methods, Task 2 `_owner_only`, Task 3 `RunStore.cancel_all`.
- Produces (all JSON, `_own_page` on POST):
  - `GET /keys` (owner) -> `{"keys": KeyStore.view(), "log": last 100 entries}`.
  - `POST /keys` (owner) body `{"action": "create", "name", "role"}` -> `{"key": "<plain, once>"}`;
    `{"action": "rename", "name", "to"}`; `{"action": "role", "name", "role"}`; `{"action": "revoke", "name"}` ->
    also `RunStore.cancel_all(name)` (part 2 adds runners).
  - `GET /devices/pending` -> the pending devices the caller `may_approve` (owner: all; a guest from an approved
    device: its own key's).
  - `POST /devices` body `{"key", "device", "approve": bool}` or `{"key", "device", "action": "clear"}` ->
    `may_approve` decides; 403 otherwise.
- The legacy owner key cannot be renamed or revoked here (it lives in the run config); the answer says so.

- [ ] **Step 1: Write the failing tests:** `test_owner_creates_and_lists_keys_without_secrets`;
  `test_guest_cannot_manage_keys`; `test_guest_approves_own_second_device` (ann approved on d1 approves d2; cannot
  approve bob's device; cannot approve her own first device); `test_revoke_cancels_runs_and_refuses_next_request`
  (RF 3); `test_approvals_are_logged`.
- [ ] **Step 2: Run** `python -m pytest serve/test_identity_http.py -q` - Expected: FAIL.
- [ ] **Step 3: Implement** the routes.
- [ ] **Step 4: Run** `python -m pytest serve -q` - Expected: all pass (full serve suite once here).
- [ ] **Step 5: Commit** `feat(serve): keys and devices managed live (#310)`; BLUEPRINT: the four routes.

### Task 6: Web app - device id, waiting screen, keys and devices, approval notice, Monitor names

**Files:**
- Modify: `serve/ui/src/lib/api.ts` (`apiHeaders`), Create: `serve/ui/src/lib/device.ts` + `device.test.ts`
- Create: `serve/ui/src/components/KeysSettings.tsx`; Modify: `serve/ui/src/pages/Settings.tsx` (section dispatch,
  near line 98)
- Create: `serve/ui/src/components/DeviceGate.tsx` (waiting screen) and `ApprovalNotice.tsx`; mount both in the app
  shell (`serve/ui/src/App.tsx` or the router root, whichever renders every page)
- Modify: `serve/ui/src/pages/Live.tsx`, `Requests.tsx` (show `who`, holder, waiting list)
- Modify: `serve/ui/src/lib/i18n*` (Thai for every new string), `serve/ui/dist/` (rebuilt)

**Interfaces:**
- Consumes: the routes of Tasks 2, 4, 5 and the 403 types.
- Produces: `deviceId(): string` (creates once, `strata.device`, 22 chars `[A-Za-z0-9_-]`), `deviceName(): string`
  (`navigator.userAgent` summary, editable in Settings, `strata.deviceName`); `apiHeaders()` adds
  `X-Strata-Device` and `X-Strata-Device-Name`; `isDevicePending(r: Response): Promise<string | null>` (the code).

- [ ] **Step 1: Write the failing tests** (`bun test`): `device.test.ts` - `deviceId()` is stable across calls and
  matches `/^[A-Za-z0-9_-]{22}$/`; `apiHeaders()` includes both device headers; `isDevicePending` returns the code for
  a 403 `device_pending` body and null for other 403s.
- [ ] **Step 2: Run** `cd serve/ui && bun test` - Expected: FAIL.
- [ ] **Step 3: Implement** the lib, then the components:
  - DeviceGate: full screen "This device waits for approval - code ABCD. Ask the workstation's owner, or approve it
    from a device you already use." and polls `GET /health` + `/props` every 5 s until it passes.
  - ApprovalNotice: polls `GET /devices/pending` every 10 s for an owner or a guest on an approved device; a banner
    per device: key, label, address, code, Approve / Refuse.
  - KeysSettings (owner only): table of keys (name, role, created, devices with status, last seen), create (shows the
    key once with copy), rename, role, revoke (confirm), clear device; the log.
  - Live/Requests: a "who" column; Live shows "in use by NAME" and "waiting: NAME (1), NAME (2)".
- [ ] **Step 4: Run** `cd serve/ui && bun test && bun run typecheck && bun run build`, then `python -m pytest serve/test_ui.py -q` - Expected: all pass.
- [ ] **Step 5: Commit** `feat(ui): device approval, keys and devices, who holds the engine (#310)` including `dist/`.

### Task 7: End-to-end check on the workstation and the record

**Files:**
- Modify: `docs/BLUEPRINT.md` (revision log row; "Not yet in the baseline" until deployed)
- No product code.

- [ ] **Step 1:** Start a second server from the branch on a free port with the mock engine
  (`python serve/server.py --engine mock --port 8099` plus a temp `--config` naming a temp `keys_file`), in the
  background with no window. Create a guest key over `POST /keys` from loopback.
- [ ] **Step 2:** From this PC through the Radmin address (`http://26.33.142.160:8099`), with the guest key and a new
  device header: expect `device_pending`; approve via `POST /devices` as owner; repeat -> 200. Record each status.
- [ ] **Step 3:** Stop the test server (`taskkill /T` its tree; verify the port is free).
- [ ] **Step 4:** Comment on #310 (English + Thai): what landed, the commits, the test counts, the Step 2 log, and that
  deploying to the served tree (`C:\Strata-exp\src-main`) and handing out keys is the developer's step.
- [ ] **Step 5: Commit** the BLUEPRINT revision row.
