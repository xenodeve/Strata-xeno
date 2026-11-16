# Strata Desktop, the program (window, tray, runner inside) - Implementation Plan (part 3 of Strata Desktop)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Windows program, `StrataDesktop.exe`, that keeps a list of Strata servers (name, URL, token), tests one,
runs this PC's runner for it inside the app, and shows that server's own web app in a window where the page and the
runner are one device and the page knows its runner id - with a tray icon that keeps the runner going while the window
is closed.

**Architecture:** A new top-level package `desktop/`. The parts that decide things are plain Python with no GUI import
(paths, DPAPI, the server list, the http warning, the server test, the injection script, the runner host, the tray
state machine, the controller that orders them); only `desktop/gui.py` imports pywebview and pystray, lazily. The runner
is `serve.runner`'s own `Client` + `Executor` + `RunnerLoop` composed in a thread (never `runner.main()`), with two
small opt-in hooks added to `serve/runner.py` (a state callback and a device label). Two pywebview windows: a local
"servers" window that has the Python API, and the "app" window on the server's URL that has none.

**Tech Stack:** Python 3.11 (`C:\Program Files\Python311`), pywebview (WebView2 / `edgechromium` backend only),
pystray + Pillow (tray icon), PyInstaller (one-folder build), ctypes for DPAPI / named mutex / named event / message
box; pytest.

**Spec:** `docs/superpowers/specs/2026-11-15-strata-desktop-design.md` - "Strata Desktop (the program)", "The web app
(serve/ui)" (Knowing its runner), "Security summary", decision 1 (device approval, the owner's tray notice), "Changes
during part 1". Parts 1 and 2 as built on branch `xeno/team-keys`: `serve/runner.py` (`runner_dir`, `load_device`,
`Client`, `Executor`, `RunnerLoop`, `EXIT_REFUSED`, `Refused` kinds), `serve/identity.py` (`device_of`, `DEVICE_ID`,
`pending()` rows), `serve/ui/src/lib/device.ts` (`ID = /^[A-Za-z0-9_-]{22}$/`), `lib/runner.ts` (`RUNNER_RE`),
`lib/store.ts` (every `strata.*` value JSON-encoded).

## Global Constraints

- Work on the branch the controller names (parts 1-2 are on `xeno/team-keys`). Never touch port 8091,
  `C:\Strata-exp\src-main` or the `D:/Github/Strata` configs. Test servers use port 8099 or 0. Issue comments are the
  controller's job: no `gh` step here.
- No window on the developer's display from any test or check. Unit tests open no window. The exe is run by tests only
  with `--selftest` (hidden, see Task 6); the visible window is the developer's check, said so in the record.
- The GUI libraries go into a project-local venv `desktop\.venv` (created by Task 1), never the global Python.
  `python -m pytest desktop -q` passes with the global Python too: no module but `desktop/gui.py` imports `webview`,
  `pystray` or `PIL`, and it imports them inside functions.
- Opt-in and byte-identical: the two hooks in `serve/runner.py` default to today's behaviour (`on_state=None`,
  `label=None`); `python -m serve.runner` prints the same lines and sends the same headers as before.
- Files: `data_dir()` = `%LOCALAPPDATA%\StrataDesktop` (`~/.strata-desktop` elsewhere), overridable by `--data-dir`.
  Inside it: `servers.json`, `webview\` (the WebView2 profile), `desktop.log` (rotated, 1 MB x 3). The runner keeps
  `serve.runner.runner_dir()` (`%LOCALAPPDATA%\StrataRunner`: `device.json`, `runner.json`, `checkpoints\`), so the app
  and `python -m serve.runner` on one PC are one device; with `--data-dir D` the runner's folder is `D\runner` (the
  selftest never touches the real device).
- `servers.json`: `{"version": 1, "last": name | null, "injected": origin | null, "servers": [{"name", "url",
  "token_dpapi": base64}]}`, written to a `.tmp` then `os.replace`. Name 1-64 characters, unique; URL `http(s)://host[:port][/path]`
  stored without a trailing `/`; token must match `serve.runner.KEY`. DPAPI: `CryptProtectData` /
  `CryptUnprotectData` with `CRYPTPROTECT_UI_FORBIDDEN` (0x1) and optional entropy `b"StrataDesktop/1"`, current user
  scope.
- The token never appears in: `servers.json` text, `desktop.log`, any `repr`/`str` of an app object, argv, the
  environment of the app or of anything the runner starts, or any value a `js_api` method returns. The selftest takes
  its token from `STRATA_DESKTOP_SELFTEST_TOKEN` and pops it at once.
- **Token in the web view's storage (a conflict with "never in plain text", decided here):** the web app reads its key
  from `localStorage` `strata.apikey` (`api.ts`), and the WebView2 profile must persist (`webview.start(private_mode=False,
  storage_path=<data_dir>\webview)`) or every chat (`strata.chats`) is lost on restart. So while a server is open its
  token sits in that profile on disk, as it does in a browser. Bounds: one origin holds it at a time; it is removed on
  Switch server, on Quit and when that server is removed; `servers.json` `"injected"` names the origin that holds it, and
  the next start removes it there before anything else (a crash leaves it until then). LevelDB may keep a removed value
  in its log until compaction. The tray/servers window never say "never stored": the servers window says "Your token is
  kept encrypted for your Windows user; while a server is open the web app holds it as a browser would."
- Device: the app uses `load_device(runner folder)` only (22 characters, the page's `ID` rule); never `--device-id`. The
  page, the server test and the runner send the same `X-Strata-Device` and the same name, `Strata Desktop on
  <machine>` (`serve.runner.device_label(device, "Strata Desktop")`).
- Injection (pywebview has no init-script API; documented calls only): the hidden app window loads
  `<base>/health` (same origin, no app script, open without key or device approval), `evaluate_js(inject_script(...))`
  must return `true`, then `load_url(<base>/)`, then `show()` on its `loaded` event. Values are JSON-encoded as
  `store.set` writes them: `strata.apikey`, `strata.runner`, `strata.device`, `strata.deviceName`. Never inject on the
  app page itself: by then it has sent its own id.
- Order of a connect: test -> runner hello -> runner id -> inject -> navigate. Nothing is injected before the runner
  is online, except "Open without My PC" (below), which injects without `strata.runner` (and removes it).
- Threads: pywebview's loop on the main thread; pystray on its own thread (`Icon.run_detached()` if Task 1's spike
  shows it works with pywebview on the main thread, else `Icon.run` in a daemon thread `desktop-tray`); the runner host
  `desktop-runner` (builds the Executor, runs `RunnerLoop.run`; the loop's own `runner-job` threads under it); the
  second-instance listener `desktop-show`; the pending-device poll `desktop-pending` (30 s). js_api calls run on
  pywebview's own threads and only call the controller.
- Words: one table `desktop/words.py`, English and Thai, Thai chosen when `GetUserDefaultUILanguage() & 0x3FF == 0x1E`.
  Exact copy where this plan gives it.
- PyInstaller: one-folder, windowed (`console=False`), `dist\StrataDesktop\StrataDesktop.exe` (one-file would unpack to
  `%TEMP%` on every start: slower start, antivirus noise). Build and venv output (`desktop/.venv/`, `desktop/build/`,
  `desktop/dist/`) are git-ignored.
- `docs/BLUEPRINT.md` in the same commit as each task: a row "#310 part 3 task N" under "Not yet in the baseline"
  (`tools/hooks/commit-msg` refuses new source files otherwise).
- Python tests: `python -m pytest desktop -q` and `desktop\.venv\Scripts\python -m pytest desktop -q` from the repo
  root; `serve` changes also `python -m pytest serve/test_runner.py serve/test_runner_state.py -q`.

## Review Focus

1. **The token leaks to disk, logs or a crash report.** `servers.json` holds only the DPAPI blob; `ServerEntry`'s
   `repr` shows `***`; the log filter blanks every token the app holds; unhandled exceptions go through
   `sys.excepthook` / `threading.excepthook` into the scrubbed log with no locals; `js_api` returns no token. A Windows
   Error Reporting dump of the process holds the token in memory, as any program's would: said in the record, not
   claimed away. Tests in Task 1 (`test_file_has_no_token`, `test_entry_repr_hides_token`), Task 2
   (`test_api_never_returns_token`), Task 4 (`test_log_scrubs_token`, `test_excepthook_scrubs`).
2. **The page and the runner disagree on the device, so approval loops** (the owner approves one, the other keeps
   asking). One id from `load_device`, 22 characters, sent by the test, the runner and the page with one label. Tests
   in Task 2 (`test_probe_sends_runner_device_headers`), Task 4 (`test_rejects_device_id_not_22`,
   `test_values_are_json_encoded`), Task 6 (the key has exactly one device row after the selftest).
3. **The runner survives Quit or Switch server** (jobs keep running, a background shell or MCP server stays, two
   runners poll). Quit and Switch call `RunnerHost.stop(5.0)`: `RunnerLoop.finish(5.0)` (cancels jobs; part 2 #314 job
   objects end children) then `Executor.close()`, then join; a 10 s watchdog ends the process if anything hangs. Tests
   in Task 3 (`test_stop_ends_threads_and_closes_executor`, `test_stop_while_executor_still_building`) and Task 5
   (`test_switch_stops_old_runner_first`, `test_quit_order`).
4. **The app starts twice** (two clicks, autostart plus a click): two runners on one device replace each other
   (`runner_replaced`) forever. A named mutex `Local\StrataDesktop` (or `Local\StrataDesktop-<sha8 of data_dir>` with
   `--data-dir`); the second process sets the event `Local\StrataDesktop.show` and exits 0, the first shows its window.
   A standalone `python -m serve.runner` with the same device takes the runner's place: the tray says so in words and
   does not retry. Tests in Task 4 (`test_second_acquire_fails`, `test_show_event_reaches_first`) and Task 3
   (`test_refused_replaced_is_terminal`).
5. **WebView2 is missing (a Windows 10 PC).** Detected before the GUI starts, by the method Microsoft documents
   (verified in Task 1's spike); missing -> a native message box with the words below and exit code 4, never a
   fallback to the MSHTML backend (`gui="edgechromium"` is forced). pywebview's own start error is caught and gives the
   same box. Test in Task 4 (`test_missing_webview2_message`, with the detector patched).

---

### Task 1: The desktop package, its venv, the spike, DPAPI and the server list

**Files:**
- Create: `desktop/__init__.py`, `desktop/requirements.txt` (pywebview, pystray, Pillow, exact `==` versions as
  installed), `desktop/requirements-build.txt` (pyinstaller, pytest, `==` versions), `desktop/paths.py`,
  `desktop/dpapi.py`, `desktop/servers.py` (the spike's findings go into this task's BLUEPRINT row and, in Task 4,
  the docstring of `desktop/gui.py`; no separate file)
- Test: `desktop/test_dpapi.py`, `desktop/test_servers.py`
- Modify: `.gitignore` (`/desktop/.venv/`, `/desktop/build/`, `/desktop/dist/`), `docs/BLUEPRINT.md`

**Interfaces:**
- Produces:
  - `paths.data_dir(override: str | None = None) -> Path`; `paths.runner_dir(data: Path, override_used: bool) -> Path`
    (`serve.runner.runner_dir()` unless `override_used`, then `data / "runner"`); `paths.servers_file(data) -> Path`,
    `paths.webview_dir(data) -> Path`, `paths.log_file(data) -> Path`.
  - `dpapi.protect(data: bytes) -> bytes`, `dpapi.unprotect(blob: bytes) -> bytes`; `class DpapiError(Exception)`
    (a blob that does not decrypt for this user, or a non-Windows system: "the token store needs Windows").
  - `class Secret`: wraps a str; `reveal() -> str`; `__repr__` / `__str__` -> `"***"`; `__eq__` by value.
  - `@dataclass(frozen=True) ServerEntry(name: str, url: str, token: Secret)`.
  - `class ServerStore(path: Path)`: `entries() -> list[ServerEntry]`; `get(name) -> ServerEntry | None`;
    `save(name: str, url: str, token: str | None, old_name: str | None = None) -> ServerEntry` (token None keeps the
    stored one; raises `ValueError` with the field's words); `remove(name) -> bool`; `last` / `set_last(name)`;
    `injected` / `set_injected(origin: str | None)`. A blob that no longer decrypts (profile moved to another user)
    gives that entry `token = Secret("")` and `public()` marks it `"token": "unreadable"`.
  - `ServerStore.public() -> list[dict]`: `{"name", "url", "token": "set" | "missing" | "unreadable"}`.

- [ ] **Step 1: Make the venv.** `python -m venv desktop\.venv`, then `desktop\.venv\Scripts\python -m pip install
  pywebview pystray Pillow pyinstaller pytest`; write the installed versions with `==` into the two requirements files.
  Expected: `desktop\.venv\Scripts\python -c "import webview, pystray, PIL"` exits 0; the global `python -c "import
  webview"` still fails (nothing installed globally).
- [ ] **Step 2: Spike (no visible window; a script in the scratchpad, not committed).** With the venv's python, in the
  background, start `python -m serve.server --engine mock --port 8099` and a pywebview script that creates one window
  with `hidden=True` on `http://127.0.0.1:8099/health`, `webview.start(gui="edgechromium", private_mode=False,
  storage_path=<tmp>)`, and from the start function: waits for `loaded`, `evaluate_js` sets `localStorage["strata.x"]`,
  `load_url(".../")`, waits for `loaded`, reads it back and `document.getElementById("root").childElementCount`, writes
  the results to a file, `destroy()`s. Also check: `events.closing` returning False keeps a hidden window;
  `pystray.Icon.run_detached()` alongside (icon only, no notification); the WebView2 detection method from Microsoft
  Learn's "Detect if a WebView2 Runtime is already installed" (record the exact registry keys / API it names). Record
  each finding as a yes/no with the command. **Fallbacks, decided now:** if a hidden window does not load or run
  script, the selftest drops its window part (Task 6 says "window: not checked") and the developer's visible check
  covers it; if `run_detached` does not work, pystray runs `Icon.run()` in thread `desktop-tray`. Stop both processes
  (`taskkill /T`) and check 8099 is free.
- [ ] **Step 3: Write the failing tests.** `test_dpapi.py` (skip off Windows): `test_round_trip` (bytes with NUL and
  non-ASCII come back equal); `test_tampered_blob_raises` (one byte flipped -> `DpapiError`); `test_blob_is_not_plaintext`
  (the token's bytes are not in the blob). `test_servers.py`: `test_save_and_reload` (a new `ServerStore` on the same
  path reads name, url, token); `test_file_has_no_token` (RF 1: the file's bytes contain neither the token nor its
  base64); `test_entry_repr_hides_token` (RF 1: `repr(entry)`, `str(entry)`, `f"{entry}"` contain `***` and not the
  token); `test_url_normalised` (`"http://26.1.2.3:8095/"` -> stored `"http://26.1.2.3:8095"`); `test_bad_fields`
  (empty name, 65-char name, duplicate name, `ftp://x`, token with a space -> `ValueError` naming the field);
  `test_save_without_token_keeps_it`; `test_rename_keeps_token` (`old_name`); `test_unreadable_blob_marked`
  (`dpapi.unprotect` patched to raise -> `public()` says `"unreadable"`, no exception); `test_public_has_no_token`;
  `test_injected_round_trip`; `test_atomic_write` (`os.replace` patched to raise -> the old file is intact).
- [ ] **Step 4: Run** `python -m pytest desktop -q` - Expected: FAIL (`ModuleNotFoundError: desktop.dpapi`).
- [ ] **Step 5: Implement** `paths.py`, `dpapi.py` (ctypes `crypt32.CryptProtectData` / `CryptUnprotectData`,
  `DATA_BLOB`, `kernel32.LocalFree` on the output buffer), `servers.py`.
- [ ] **Step 6: Run** `python -m pytest desktop -q` and `desktop\.venv\Scripts\python -m pytest desktop -q` - Expected: all pass.
- [ ] **Step 7: Commit** `feat(desktop): the server list with DPAPI tokens, the desktop venv (#310 part 3)`; BLUEPRINT
  row "#310 part 3 task 1": `desktop/` and its files, `servers.json` format, the spike's yes/no findings.

### Task 2: The http warning and the server test

**Files:**
- Create: `desktop/urlcheck.py`, `desktop/probe.py`, `desktop/words.py`
- Modify: `serve/runner.py` (`device_label`, `Client(label=)`)
- Test: `desktop/test_urlcheck.py`, `desktop/test_probe.py`, `desktop/test_words.py`, `serve/test_runner_state.py`
  (new; the label tests here, the state tests in Task 3)

**Interfaces:**
- Produces:
  - `serve.runner.device_label(device: dict, app: str = "Strata runner") -> str` = `_header(f"{app} on {device.get('name')
    or machine_name()}")`; `Client.__init__(server, key, device, label: str | None = None)` sends `label` when given,
    else `device_label(device)` (today's header, unchanged).
  - `urlcheck.plain_http_warning(url: str) -> str | None`: None for `https`; None for `http` to `localhost`, a
    loopback address (127.0.0.0/8, ::1), 26.0.0.0/8 (Radmin) or 100.64.0.0/10 (Tailscale), or a host name ending
    `.ts.net` (Tailscale MagicDNS); otherwise `words.t("plain_http")`. No DNS lookup (the rule is the URL's text). A
    private LAN address (192.168.x) over http also warns: the token travels in clear there too.
  - `probe.device_headers(device: dict, label: str) -> dict[str, str]` (`X-Strata-Device`, `X-Strata-Device-Name`);
    `@dataclass ProbeResult(reachable: bool, model: str | None, role: "owner" | "guest" | None, pending_code: str |
    None, engine_running: bool, error: str | None)`; `probe.test_server(url: str, token: str, device: dict, label:
    str, timeout: float = 10.0) -> ProbeResult`: GET `/health` (no key; 200 or 503 `engine_exited` = reachable, model
    from the body); GET `/props` with `Authorization: Bearer` + device headers (401 -> error `words.t("token_refused")`;
    403 `device_pending` -> `pending_code`; 403 `device_refused` -> error `words.t("device_refused")`; 503 ->
    `engine_running=False`, go on); then GET `/keys` with the same headers: 200 -> owner, 403 -> guest, anything else ->
    role None. Every request of the test uses the same headers (RF 2).
  - `words.t(key: str, **kw) -> str`, `words.LANG` ("en" | "th"), `words.TABLE: dict[str, dict[str, str]]`. Exact copy:
    - `plain_http`: "This address is plain http to another network: your token and the files your PC sends travel
      unencrypted. Use Tailscale, Radmin VPN or an https tunnel." / "ที่อยู่นี้เป็น http ธรรมดาไปยังเครือข่ายอื่น
      โทเค็นของคุณและไฟล์ที่เครื่องคุณส่งไปจะวิ่งแบบไม่เข้ารหัส ใช้ Tailscale, Radmin VPN หรือ tunnel แบบ https แทน"
    - `pending`: "Waiting for approval on {server}: code {code}" / "รออนุมัติที่ {server}: รหัส {code}"
    - `update`: "Update Strata Desktop: {server} speaks runner protocol {n}." / "อัปเดต Strata Desktop: {server}
      ใช้โปรโตคอล runner รุ่น {n}"
    - `no_webview2`: "Strata Desktop needs the Microsoft Edge WebView2 Runtime, which this PC does not have. Install
      it from https://go.microsoft.com/fwlink/p/?LinkId=2124703 and start Strata Desktop again." / Thai mirror.
    - the tray words of Task 3 and the servers window's labels; the implementer writes the Thai for those.

- [ ] **Step 1: Write the failing tests.** `test_urlcheck.py`: `test_https_never_warns`; `test_quiet_cases`
  (`http://127.0.0.1:8095`, `http://localhost:8095`, `http://[::1]:8095`, `http://26.33.142.160:8095`,
  `http://100.64.0.1`, `http://100.127.255.254`, `http://strata.tail1234.ts.net` -> None);
  `test_warn_cases` (`http://192.168.1.10:8095`, `http://100.128.0.1`, `http://25.255.255.255`,
  `http://example.com` -> the `plain_http` text). `test_probe.py` (a scripted `http.server` on port 0 in a thread):
  `test_owner` (health 200, props 200 `model_alias`, keys 200 -> role owner, model); `test_guest` (keys 403 ->
  guest); `test_token_refused` (props 401); `test_pending_code` (props 403 `{"error": {"type": "device_pending",
  "code": "K7QZ"}}` -> `pending_code == "K7QZ"`, no `/keys` request); `test_engine_down_still_reports_role` (health
  503, props 503, keys 200); `test_unreachable` (closed port -> `reachable False`, error text, within 11 s);
  `test_probe_sends_runner_device_headers` (RF 2: every request after `/health` carried `X-Strata-Device ==
  device["id"]` and `X-Strata-Device-Name == "Strata Desktop on ann-pc"`). `test_words.py`:
  `test_every_key_has_thai`; `test_format_keys_match` (the `{}` fields of the two languages are equal).
  `serve/test_runner_state.py`: `test_label_default_unchanged` (`Client(url, key, {"id": "x"*22, "name": "pc"})`
  header == `"Strata runner on pc"`); `test_label_given` (`label="Strata Desktop on pc"` is sent).
- [ ] **Step 2: Run** `python -m pytest desktop serve/test_runner_state.py -q` - Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the same plus `python -m pytest serve/test_runner.py -q` - Expected: all pass.
- [ ] **Step 5: Commit** `feat(desktop): the server test (health, props, role) and the plain-http warning (#310 part 3)`;
  BLUEPRINT row: `Client(label=)`, `device_label`.

### Task 3: The runner inside the app, and the tray's state machine

**Files:**
- Modify: `serve/runner.py` (`RunnerLoop(on_state=)`)
- Create: `desktop/runner_host.py`, `desktop/state.py`
- Test: `serve/test_runner_state.py`, `desktop/test_runner_host.py`, `desktop/test_state.py`

**Interfaces:**
- Produces:
  - `RunnerLoop.__init__(client, executor, *, stop=None, on_state: Callable[[str, dict], None] | None = None)`. Calls
    (each wrapped so a raising callback is logged and ignored): `("online", {"id", "who"})` after every hello that
    succeeds and when a poll succeeds after network failures; `("pending", {"code"})` on `Pending`; `("offline",
    {"message"})` (scrubbed) on the first `NetworkError` of a run of failures; `("refused", {"kind", "message"})`
    before returning `EXIT_REFUSED`; `("stopped", {})` when `run` returns 0. No other behaviour changes.
  - `class RunnerHost(url: str, token: str, device: dict, label: str, runner_folder: Path, on_state: Callable[[str,
    dict], None], executor_factory: Callable[[], object] | None = None)`: `start() -> None` (thread `desktop-runner`:
    calls `on_state("connecting", {})`, builds `Executor(harness.home_dir(), runner_folder / "runner.json",
    runner_folder)` - a `SystemExit` or any exception -> `on_state("error", {"message"})` and the thread ends - then
    `RunnerLoop(...).run()`); `stop(timeout: float = 5.0) -> bool` (sets stop; `loop.finish(timeout)`;
    `executor.close()`; joins; True when the thread ended; safe before the executor exists and twice);
    `runner_id: str | None`, `who: str | None`, `shell: str | None` (from `executor.info()`). `serve.runner._say` is
    routed to the desktop log by Task 4; this class never prints.
  - `state.py`: `@dataclass(frozen=True) TrayState(kind: Literal["idle", "connecting", "online", "pending", "offline",
    "refused", "error"], server: str = "", who: str = "", runner_id: str = "", code: str = "", refused: str = "",
    message: str = "", no_shell: bool = False)`; `step(s: TrayState, event: str, data: dict) -> TrayState`;
    `tooltip(s) -> str` (at most 127 characters: the server name is cut first); `icon_color(s) -> "green" | "amber" |
    "grey" | "red"`; `menu(s) -> list[str]` (always `["open", "switch", "quit"]`; `"copy_code"` added while pending).
  - Transitions: `connecting` -> `connecting`; `online` -> `online` (who, id); `pending` -> `pending` (code);
    `offline` -> `offline` (keeps who/id); `refused` -> `refused` (terminal: later events but `connecting` are
    ignored); `error` -> `error` (terminal likewise); `stopped` -> `idle`. Tooltip words (English; Thai in `words`):
    online "Online - {server} as {who}", connecting "Connecting to {server}…", pending = `words.t("pending")`,
    offline "Offline - {server} cannot be reached; trying again", refused by kind: `key` "The key for {server} was
    refused (wrong or revoked)", `device_refused` "The owner of {server} refused this PC", `runner_replaced` "Another
    Strata runner on this PC took over", `runner_protocol` = `words.t("update", n=<from the message>)`, `refused`
    "{server} refused the runner"; error "My PC's tools could not start: {message}"; idle "Strata Desktop - not
    connected". `no_shell` (the runner has no shell: `shell` is None) appends " (no shell: Git for Windows was not
    found, Bash is off)".

- [ ] **Step 1: Write the failing tests.** `serve/test_runner_state.py` (a scripted fake client, no HTTP, a fake
  executor with `groups/skills/info/starting`): `test_states_in_order` (hello raises `Pending("K7QZ")`, then succeeds,
  `next` raises `NetworkError`, then returns `[]`, then `Refused("key", "")` -> the callback saw `pending K7QZ`,
  `online`, `offline`, `online`, `refused key`, and `run()` returned 3); `test_stop_reports_stopped`;
  `test_callback_error_is_ignored` (a raising callback does not end the loop); `test_no_callback_unchanged` (`_say`
  lines with `on_state=None` equal those of the same script before the change). `desktop/test_runner_host.py`:
  `test_connects_to_real_server` (an in-process `serve.server.serve(svc, port=0)` with `api_key`, no team mode, as
  `serve/test_runner.py` builds one; `runner_folder` a temp dir -> state `online` within 10 s and `runner_id ==
  runner_hub.runner_id("owner", device["id"])`); `test_stop_ends_threads_and_closes_executor` (RF 3: after
  `stop(5.0)` returns True, no thread named `desktop-runner`, `runner-poll` or `runner-job` is alive and the fake
  executor's `close` was called once); `test_stop_while_executor_still_building` (RF 3: a factory that blocks 1 s ->
  `stop` returns, `close` is called once the build ends, no runner loop starts); `test_bad_runner_json_is_error`
  (`runner.json` = `[1]` -> state `error` with the `must be a JSON object` text, no exception escapes);
  `test_refused_replaced_is_terminal` (RF 4: a second host with the same device on the same server -> the first gets
  `refused runner_replaced` and its thread ends). `desktop/test_state.py`: one test per transition above;
  `test_refused_is_terminal`; `test_tooltip_fits_128` (a 64-character server name and a 64-character who ->
  `len <= 127`); `test_protocol_words` (`refused runner_protocol` with the server's 409 message "update Strata Desktop
  (the server speaks runner protocol 2)" -> tooltip contains "protocol 2").
- [ ] **Step 2: Run** `python -m pytest serve/test_runner_state.py desktop -q` - Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the same plus `python -m pytest serve/test_runner.py -q` - Expected: all pass.
- [ ] **Step 5: Commit** `feat(desktop): the runner in a thread of the app, its states for the tray (#310 part 3)`;
  BLUEPRINT row: `RunnerLoop(on_state=)`, the thread `desktop-runner`.

### Task 4: Injection, one instance, the log, WebView2, and the windows

**Files:**
- Create: `desktop/inject.py`, `desktop/single.py`, `desktop/applog.py`, `desktop/webview2.py`, `desktop/gui.py`,
  `desktop/ui/servers.html` (the first screen: plain HTML/CSS/JS, no CDN, light and dark), `desktop/ui/icon.png`
- Test: `desktop/test_inject.py`, `desktop/test_single.py`, `desktop/test_applog.py`, `desktop/test_webview2.py`,
  `desktop/test_imports.py`

**Interfaces:**
- Produces:
  - `inject.storage_values(token: str, runner_id: str | None, device: dict, label: str) -> tuple[dict[str, str],
    list[str]]` -> (`{"strata.apikey": json.dumps(token), "strata.device": json.dumps(device["id"]),
    "strata.deviceName": json.dumps(label)} + {"strata.runner": json.dumps(runner_id)} when given`, keys to remove
    (`["strata.runner"]` when None)); raises `ValueError` when `device["id"]` is not `^[A-Za-z0-9_-]{22}$` or
    `runner_id` not `^r[0-9a-f]{16}$` (the page would drop either and make its own).
  - `inject.inject_script(values: dict[str, str], remove: list[str]) -> str`: one JS expression that sets and removes
    them and evaluates to `true` only when every value reads back equal and every removed key is absent. The values go
    in as one `json.dumps(..., ensure_ascii=True)` literal (U+2028 and `</script>` safe).
  - `inject.clear_script() -> str` (removes `strata.apikey`, evaluates to `true` when it is gone);
    `inject.origin(url) -> str`; `inject.health_url(base) -> str` (`base + "/health"`); `inject.app_url(base) -> str`
    (`base + "/"`).
  - `single.acquire(name: str) -> Instance | None` (`CreateMutexW`; None when `GetLastError() == 183`);
    `Instance.listen(on_show: Callable[[], None]) -> None` (thread `desktop-show` waiting on event `name + ".show"`);
    `single.signal_show(name: str) -> bool`; `single.instance_name(data_dir: Path, overridden: bool) -> str`.
  - `applog.setup(path: Path) -> logging.Logger` (rotating 1 MB x 3); `applog.add_secret(s: str)`; a filter that
    replaces every added secret with `***` in the message and in formatted exceptions; `applog.install_hooks()`
    (`sys.excepthook`, `threading.excepthook`: the traceback's text only, scrubbed, no locals); `applog.say(text)` - what
    `serve.runner._say` is set to.
  - `webview2.installed() -> str | None` (the runtime's version, by the method Task 1's spike recorded);
    `webview2.missing_message() -> str` (`words.t("no_webview2")`); `webview2.message_box(text: str) -> None`
    (`MessageBoxW`, `MB_ICONERROR`); `webview2.preflight() -> int | None` (None when installed; else shows the box and
    returns 4).
  - `gui.py` (the only module importing `webview` / `pystray` / `PIL`, inside functions): `class Gui` implementing the
    controller's `WindowPort` (Task 5): `servers` window (`html=` the contents of `ui/servers.html`, `js_api=ServersApi`,
    320 x 520 minimum) and `app` window (`hidden=True`, no `js_api`, title "Strata - {server}"); `closing` on either
    returns False and hides it unless the controller is quitting; `webview.start(func, gui="edgechromium",
    private_mode=False, storage_path=str(paths.webview_dir(data)))`; `settings["ALLOW_DOWNLOADS"] = True`.
    `navigate_and_inject(base: str, script: str, timeout: float = 15.0) -> bool` (load `health_url`, wait `loaded`,
    `evaluate_js(script)`, True only on `True`) and `open_app(base: str)` (load `app_url`, show on `loaded`).
  - `ServersApi` (methods the servers page calls; none returns a token, RF 1): `list() -> {"servers": public(), "last",
    "state": <TrayState as dict>}`; `warning(url) -> str | None`; `save(name, url, token | None, old_name | None) ->
    {"ok"} | {"error"}`; `remove(name)`; `test(name) -> ProbeResult as dict` (role, model, pending code, error);
    `connect(name)`; `open_without_runner()`; `copy_code()`.

- [ ] **Step 1: Write the failing tests.** `test_inject.py`: `test_values_are_json_encoded` (RF 2: `values
  ["strata.device"] == '"' + id + '"'`, the token `ab"c\u2028</script>` comes back from `json.loads`);
  `test_without_runner_removes_runner_key`; `test_rejects_device_id_not_22` (RF 2: 21 and 23 characters, `+` ->
  `ValueError`); `test_rejects_bad_runner_id`; `test_script_runs_in_node` (skip when `node` is absent: the script run
  against a `Map`-backed `localStorage` stub evaluates to `true` and leaves exactly the values; a stub whose `setItem`
  drops `strata.apikey` -> `false`); `test_clear_script_removes_only_apikey` (node, same stub); `test_urls`
  (`base "http://h:1/p"` -> `http://h:1/p/health`, `http://h:1/p/`, origin `http://h:1`). `test_single.py` (skip off
  Windows): `test_second_acquire_fails` (RF 4: a unique name; second `acquire` -> None; after `release` -> acquired
  again); `test_show_event_reaches_first` (RF 4: `listen(cb)`, `signal_show(name)` -> cb within 2 s). `test_applog.py`:
  `test_log_scrubs_token` (RF 1: `add_secret("sk-strata-abc")`; a message, an `%s` argument and an exception text
  containing it -> the file has `***` and not the token); `test_excepthook_scrubs` (an exception raised in a thread
  with the token in its message -> logged scrubbed); `test_runner_say_goes_to_log`. `test_webview2.py`:
  `test_missing_webview2_message` (RF 5: `installed` and `message_box` patched -> `preflight()` returns 4 and the box
  got `words.t("no_webview2")`); `test_installed_reads_version` (the documented source patched to a
  version string -> returned). `test_imports.py`: `test_imports_without_gui_libs` (with `sys.modules` entries for
  `webview`, `pystray`, `PIL` set to None, importing every `desktop` module but `gui` succeeds).
- [ ] **Step 2: Run** `python -m pytest desktop -q` - Expected: FAIL.
- [ ] **Step 3: Implement.** `servers.html`: the list (name, URL, "token set", role and model after a test), the form
  (name, URL, token as a password field, the `warning` text under the URL as typed), Test, Connect, Remove; a state
  line from `list().state` refreshed by `window.render(state)` that the controller calls with `evaluate_js`; while
  pending the code in large type with "Copy code"; on `refused runner_protocol` / `error` an "Open without My PC" button.
  The token is never put back into the form.
- [ ] **Step 4: Run** `python -m pytest desktop -q` and the venv's run - Expected: all pass.
- [ ] **Step 5: Commit** `feat(desktop): injection before the app loads, one instance, a scrubbed log, the windows (#310 part 3)`;
  BLUEPRINT row: the two windows, the WebView2 profile folder, the injected values, the thread `desktop-show`.

### Task 5: The controller, the tray, pending notices, Quit and Switch

**Files:**
- Create: `desktop/controller.py`, `desktop/tray.py` (pystray adapter, imports inside functions), `desktop/__main__.py`
- Modify: `desktop/gui.py` (wires `Controller`), `desktop/ui/servers.html` (render states)
- Test: `desktop/test_controller.py`

**Interfaces:**
- Consumes: Tasks 1-4.
- Produces:
  - Ports (Protocols, so tests use fakes): `WindowPort`: `show_servers()`, `hide_servers()`, `render(state: dict)`,
    `navigate_and_inject(base, script) -> bool`, `open_app(base)`, `hide_app()`, `destroy()`; `TrayPort`:
    `update(s: TrayState)`, `notify(title: str, text: str)`, `stop()`.
  - `class Controller(store: ServerStore, device: dict, label: str, runner_folder: Path, window: WindowPort, tray:
    TrayPort, make_host: Callable[..., RunnerHost] = RunnerHost, probe=probe.test_server)`:
    - `start() -> None`: first, if `store.injected` is set, `navigate_and_inject(<that origin's base>,
      clear_script())` and `set_injected(None)` (the crash leftover); then `show_servers()`; if `store.last` names a
      server, connect to it.
    - `connect(name: str) -> None`: stop the current host first (`stop(5.0)`; clear the old origin's key, RF 3); probe;
      on `reachable False` or a token/device refusal -> render the error, no host; else start a host. On the host's
      first `online`: `navigate_and_inject(base, inject_script(storage_values(token, id, ...)))` -> `True` ->
      `set_injected(origin)`, `open_app(base)`, `hide_servers()`, `set_last(name)`; `False` -> render "the window could
      not be prepared" and keep the servers window. A later `online` with another runner id (a key rename) -> inject
      again and `open_app` (reload); the same id -> nothing. `pending` -> render the code, tray update (the runner
      itself says hello again every 10 s; nothing else polls). `refused` / `error` -> render the words; for
      `runner_protocol` and `error` the servers window offers `open_without_runner()`.
    - `open_without_runner() -> None`: inject with `runner_id=None` (removes `strata.runner`) and open the app.
    - `switch() -> None`: `hide_app()`, `show_servers()` (the runner keeps running until another server is chosen).
    - `quit() -> None`, in this order (RF 3): `quitting = True`; `host.stop(5.0)`; if a server is open,
      `navigate_and_inject(base, clear_script())` (a cap of 3 s) and `set_injected(None)`; `tray.stop()`;
      `window.destroy()`; release the mutex. A watchdog thread started at the top calls `os._exit(0)` after 10 s and
      logs which step hung.
    - `remove(name)`: when it is the open server, as quit's runner/clear steps, then `show_servers()`.
    - Pending notices (spec decision 1): thread `desktop-pending`, while a host is online, every 30 s GET
      `/devices/pending` with the token and device headers; each row not seen before -> `tray.notify("A device asks to
      be let in", "{key} on {label} ({addr}), code {code}. Open Strata Desktop, Settings > Keys, to decide.")` (Thai
      in `words`); errors are silent (logged). Deciding stays in the web app.
  - `__main__.main(argv: list[str] | None = None) -> int`: flags `--data-dir PATH`, `--selftest`, `--server URL`
    (selftest only), `--debug` (pywebview debug, never in the build's shortcut). Order: `applog.setup`, hooks,
    `serve.runner._say = applog.say`; `preflight()` (WebView2; exit 4 with the box); `single.acquire` (None ->
    `signal_show`, exit 0); load store; `device = serve.runner.load_device(runner_folder)`; `label =
    device_label(device, "Strata Desktop")`; every token read is `applog.add_secret`ed; `Gui(...).run(controller)`.
    Exit codes: 0 normal, 4 no WebView2, 5 selftest failed.

- [ ] **Step 1: Write the failing tests** (fake window, tray, host factory and probe; no GUI):
  `test_connect_waits_for_hello_then_injects` (no `navigate_and_inject` until the fake host reports `online`; then the
  script carries the runner id and the device id; `open_app` after it; `store.injected` is the origin);
  `test_pending_shows_code_then_opens` (host reports `pending K7QZ` -> `render` got the code, tray `pending`; then
  `online` -> opens); `test_runner_id_change_reinjects` (second `online` with another id -> a second inject with the
  new id; the same id -> none); `test_inject_false_keeps_servers_window`; `test_refused_key_does_not_open`;
  `test_protocol_offers_open_without_runner` (then `open_without_runner()` injects with `strata.runner` in the remove
  list); `test_switch_stops_old_runner_first` (RF 3: connect A, connect B -> A's `stop` returns before B's host is
  made, and A's origin got `clear_script()` before B's inject); `test_quit_order` (RF 3: the recorded calls are host
  stop, clear, `set_injected(None)`, tray stop, destroy - in that order); `test_quit_with_hung_stop_exits`
  (`stop` blocks; `os._exit` patched -> called within 10.5 s); `test_crash_leftover_cleared_at_start` (`injected` set
  in the store -> the first window call is the clear on that origin, then `injected` is None);
  `test_api_never_returns_token` (RF 1: every `ServersApi` method's return value, `json.dumps`ed, lacks the token);
  `test_pending_notice_once_per_device` (a fake `/devices/pending` answering the same row twice -> one `notify`).
- [ ] **Step 2: Run** `python -m pytest desktop/test_controller.py -q` - Expected: FAIL.
- [ ] **Step 3: Implement**; `tray.py`: `pystray.Icon("StrataDesktop", <icon tinted by icon_color>, tooltip(state),
  menu=Open / Switch server / Copy code (while pending) / Quit)`, Open = `show_servers()` or the app window when one is
  open (default action, double click), threading per Task 1's spike finding.
- [ ] **Step 4: Run** `python -m pytest desktop -q` and the venv's run - Expected: all pass.
- [ ] **Step 5: Commit** `feat(desktop): connect, switch and quit in order; tray states and device notices (#310 part 3)`;
  BLUEPRINT row: the process (Strata Desktop: main GUI thread, tray, `desktop-runner`, `desktop-pending`,
  `desktop-show`), its flags and exit codes, `STRATA_DESKTOP_SELFTEST_TOKEN`.

### Task 6: The selftest, the build, and the check against a mock server on 8099

**Files:**
- Create: `desktop/selftest.py`, `desktop/StrataDesktop.spec`, `desktop/build.py`, `desktop/check_selftest.py`
  (the harness: starts the server, makes the keys, approves, reads the result)
- Test: `desktop/test_selftest.py`
- Modify: `docs/BLUEPRINT.md` (the record row)

**Interfaces:**
- Produces:
  - `selftest.run(server: str, data_dir: Path, token: str, result_path: Path, window: WindowPort | None) -> int` (0
    pass, 5 fail). Steps, each recorded as `{"step", "ok", "detail", "ms"}` in `result_path` (`<data-dir>\selftest.json`;
    a windowed exe has no stdout): `store` (save the entry, reload, the token round-trips through DPAPI); `probe` (role,
    model); `runner` (a `RunnerHost` with `runner_dir = data_dir\runner`; waits up to 120 s for `online`, writing
    `{"step": "pending", "code"}` at once when pending so the harness can approve); `window` (hidden app window:
    `navigate_and_inject` true, `open_app`, then `evaluate_js` reads the four values and
    `document.getElementById("root").childElementCount > 0` and, 5 s after load, no `[data-device-gate]` element (the
    waiting screen, `DeviceGate.tsx:65`); skipped with "window: not checked" when Task 1's spike said hidden windows do
    not load); `runners` (GET `/runners` with the token and
    device headers: own runner online, its id equals the injected `strata.runner`); `quit` (the `quit()` order of Task
    5: runner threads gone, `strata.apikey` read back absent). Selftest never shows a window.
  - `StrataDesktop.spec`: `Analysis(["desktop/__main__.py"], pathex=[repo root], hiddenimports=collect_submodules
    ("serve") minus `serve.test_*`, `serve.*_fake*`, `serve.ui_hash`, `serve.dev`; datas = `desktop/ui/*`;
    `excludes=["tkinter"]`)`, `EXE(console=False, name="StrataDesktop", icon=...)`, `COLLECT` (one folder).
  - `build.py`: runs `desktop\.venv\Scripts\pyinstaller desktop\StrataDesktop.spec --noconfirm --distpath desktop\dist
    --workpath desktop\build` without a window (`CREATE_NO_WINDOW`), prints the exe's path, size and sha256.

- [ ] **Step 1: Write the failing test.** `test_selftest.py`: `test_selftest_passes_in_process` (no exe, no window:
  `window=None`; an in-process `serve()` in team mode with an owner key and a guest key `ann`; a thread approves the
  pending device with the owner key when `selftest.json` shows `pending`; `run` returns 0 and every step but `window`
  is ok; the owner's GET `/keys` shows exactly one device row for `ann` (RF 2), its id equals `device.json`'s and its
  label `Strata Desktop on <machine>`); `test_selftest_fails_on_wrong_token` (returns 5, the `probe` step says the token
  was refused, and `selftest.json` does not contain the token).
- [ ] **Step 2: Run** `python -m pytest desktop/test_selftest.py -q` - Expected: FAIL. **Step 3: Implement**
  `selftest.py`, wire `--selftest` in `__main__`. **Step 4: Run** `python -m pytest desktop -q` - Expected: all pass.
- [ ] **Step 5: Build.** `desktop\.venv\Scripts\python desktop\build.py` (background, no window). Expected: `desktop\dist\
  StrataDesktop\StrataDesktop.exe` exists; record size and sha256.
- [ ] **Step 6: Run the check, in the background.** `desktop\.venv\Scripts\python desktop\check_selftest.py --port 8099`
  does, with `CREATE_NO_WINDOW` for every child: a temp folder with a run config `{"api_key": <owner>, "keys_file":
  "keys.json"}`; `python -m serve.server --engine mock --port 8099 --host 127.0.0.1 --config <it>`; POST `/keys`
  create the first named key (owner) and a guest `ann`; starts the exe with `--selftest --server http://127.0.0.1:8099
  --data-dir <temp>\d` and `STRATA_DESKTOP_SELFTEST_TOKEN=<ann's key>` in its environment only; approves the pending
  device with the owner key when `selftest.json` shows it; waits for the exe to exit (cap 180 s); prints
  `selftest.json`, the exe's exit code and the owner's view of `ann`'s devices; ends every process tree (`taskkill /T`)
  and checks 8099 is free. Expected: exit code 0; every step ok (or `window: not checked` per the spike); one device
  row for `ann`; nothing left on 8099; the developer's display showed nothing (no step creates a visible window).
- [ ] **Step 7: Commit** `feat(desktop): --selftest, the PyInstaller build and its check on a mock server (#310 part 3)`;
  BLUEPRINT row "#310 part 3": what was checked headless (each step with its number), the exe sha256, what was not:
  the visible window and the tray on the developer's display, a Windows 10 PC without WebView2, a server over a tunnel
  - all the developer's check, with a written list in the row: open the app, connect to a test server on 8099, see the
  servers list, the warning on an http LAN address, the pending code, the chat with a My PC project, close to the tray,
  Open, Switch server, Quit (no `StrataDesktop.exe`, `bash.exe` or MCP server process left).

---

## Self-review

- Spec coverage: server list with Test (T1, T2, T4 page); the webview on the server's URL with the values injected
  (T4, T5); the runner inside the app, running while the window is closed, tray online / offline / pending / which
  server, Open / Switch / Quit (T3, T5); DPAPI tokens (T1); packaging (T6); Git Bash missing reported, not a crash
  (T3 `no_shell`, `Executor` already installs no Bash without a shell); versions - page from the server, protocol
  mismatch in words (T3 `runner_protocol`, T4 "Open without My PC"); the http warning (T2); the owner's notice of a
  pending device (T5; guests with an approved device get it for their own key, as `/devices/pending` filters); device
  pending shown and approval awaited (T3, T5); the runner's MCP servers and skills from `runner.json` and `harness.py`
  (unchanged: `Executor` does it). Left out: an editor for `runner.json` in the app (part 2 left it hand-edited; still
  so), autostart with Windows, an installer and code signing.
- Decisions beyond the spec and the controller's list: the persisted WebView2 profile and its token bounds (Global
  Constraints); one runner folder shared with `python -m serve.runner` (one device per PC); the label `Strata Desktop on
  <machine>` for both page and runner (`Client(label=)`); two windows so the remote page never gets the Python API;
  `.ts.net` names and LAN addresses in the http rule; one-folder build; the selftest writes `selftest.json` instead of
  stdout; exit codes 4 and 5; a 10 s quit watchdog; English + Thai words chosen by the Windows UI language.
