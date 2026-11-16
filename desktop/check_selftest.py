"""The check of the built StrataDesktop.exe against a mock server (#310 part 3), with no window anywhere:

    desktop\\.venv\\Scripts\\python desktop\\check_selftest.py --port 8099 [--address IP] [--exe PATH] [--out DIR]

1. A temp folder with a run config `{"api_key": <owner>, "keys_file": "keys.json", "history": {"enabled": false}}`.
2. `python -m serve.server --engine mock --port PORT --host 0.0.0.0 --config <it>` on the python the venv was made
   from (the server needs jinja2, which the venv does not have).  0.0.0.0, not 127.0.0.1: a request from this PC to
   127.0.0.1 is approved without asking and its device is never recorded (`_loopback_local`, serve/server.py), so the
   exe talks to this PC's own non-loopback address (`--address`, by default the first one found) and is a device that
   waits for the owner, as over the VPN.  The owner's calls stay on 127.0.0.1.
3. POST /keys: the first named key (owner) and a guest `ann`.
4. `<temp>\\d\\runner\\runner.json` with one MCP server (serve/mcp_fake_server.py on that python): a process the
   runner owns, which the selftest's quit step must see end with Quit (by PID).
   The exe: `--selftest --server http://ADDRESS:PORT --data-dir <temp>\\d`, with ann's key in
   STRATA_DESKTOP_SELFTEST_TOKEN and STRATA_HOME / APPDATA in the temp folder, in its environment only.
5. When selftest.json shows `pending`, the owner approves the device with that code (GET /devices/pending, POST
   /devices).  While the exe runs, its process tree and ann's runner (GET /runners) are watched; the time from the
   approval to the runner first listed online (asked every 50 ms meanwhile) is `approval_to_runner_online_s`.
6. The exe's exit (cap 180 s); then before anything is ended: the processes of the exe's tree or with the temp folder
   in their command line that are still there; the owner's view of ann's devices and runners; whether the token's
   bytes are still in the web view's profile files (LevelDB may keep a removed value until compaction: counted, not
   judged).
7. Every process tree started here ends (`taskkill /T /F` by PID, never by name), and the port is checked free.

Prints one JSON summary (never a key) and exits 0 when every check holds.  Every child is started with
CREATE_NO_WINDOW.  The temp folder is removed unless --keep; selftest.json and desktop.log are copied to --out.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NO_WINDOW = 0x08000000 if os.name == "nt" else 0
NEW_GROUP = 0x00000200 if os.name == "nt" else 0
EXE = ROOT / "desktop" / "dist" / "StrataDesktop" / "StrataDesktop.exe"
EXE_CAP_S = 180.0
TOKEN_ENV = "STRATA_DESKTOP_SELFTEST_TOKEN"


class Stop(Exception):
    """The check cannot go on (the server did not start, a key was not made): said in the summary."""


def server_python() -> Path:
    """The python the venv was made from (it has jinja2), else this one."""
    base = Path(sys.base_prefix) / ("python.exe" if os.name == "nt" else "bin/python3")
    return base if base.is_file() else Path(sys.executable)


def own_address() -> str | None:
    """This PC's first IPv4 address that is not loopback."""
    try:
        addrs = socket.gethostbyname_ex(socket.gethostname())[2]
    except OSError:
        return None
    return next((a for a in addrs if not a.startswith("127.")), None)


def listening(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def req(port: int, method: str, path: str, key: str | None, body: dict | None = None, timeout: float = 15.0):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    h = {"Authorization": "Bearer " + key} if key else {}
    if body is not None:
        h["Content-Type"] = "application/json"
    try:
        c.request(method, path, body=json.dumps(body).encode() if body is not None else None, headers=h)
        r = c.getresponse()
        raw = r.read()
        try:
            return r.status, json.loads(raw) if raw else None
        except ValueError:
            return r.status, None
    finally:
        c.close()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tree(pid: int) -> list[tuple[int, str]]:
    """`pid`'s descendants now, as (pid, name)."""
    from desktop.selftest import processes
    procs = processes()
    out, todo = [], [pid]
    while todo:
        p = todo.pop()
        for child, parent, name in procs:
            if parent == p and child != p and (child, name) not in out:
                out.append((child, name))
                todo.append(child)
    return out


def alive(pids: set[int]) -> list[tuple[int, str]]:
    from desktop.selftest import processes
    return [(p, name) for p, _, name in processes() if p in pids]


def by_command_line(text: str) -> list[dict]:
    """Processes whose command line holds `text` (Win32_Process through PowerShell, no window); this script's own
    query process excluded."""
    if os.name != "nt":
        return []
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and $_.CommandLine.Contains($env:CHECK_TEXT)"
          " -and $_.Name -ne 'powershell.exe' } | ForEach-Object { '{0}|{1}' -f $_.ProcessId, $_.Name }")
    env = {**os.environ, "CHECK_TEXT": text}
    out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], capture_output=True,
                         text=True, timeout=60, creationflags=NO_WINDOW, env=env)
    rows = []
    for line in out.stdout.splitlines():
        if "|" in line:
            pid, name = line.strip().split("|", 1)
            if pid.isdigit() and int(pid) != os.getpid():
                rows.append({"pid": int(pid), "name": name})
    return rows


def end_tree(pid: int) -> int:
    return subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True, timeout=30,
                          creationflags=NO_WINDOW).returncode


def read_result(path: Path) -> dict | None:
    for _ in range(5):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError):            # being replaced: read again
            time.sleep(0.05)
    return None


def token_in_files(folder: Path, token: str) -> list[str]:
    needles = (token.encode("utf-8"), token.encode("utf-16-le"))
    hits = []
    for p in folder.rglob("*") if folder.is_dir() else []:
        try:
            data = p.read_bytes() if p.is_file() else b""
        except OSError:
            continue
        if any(n in data for n in needles):
            hits.append(str(p.relative_to(folder)))
    return hits


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--address", help="this PC's non-loopback IPv4 address the exe uses (default: the first found)")
    ap.add_argument("--exe", default=str(EXE))
    ap.add_argument("--out", help="a folder for selftest.json, desktop.log and summary.json")
    ap.add_argument("--keep", action="store_true", help="keep the temp folder")
    a = ap.parse_args(argv)
    if a.port == 8091:
        print("check: port 8091 is the developer's server; use 8099")
        return 2
    exe = Path(a.exe)
    address = a.address or own_address()
    if not exe.is_file() or not address:
        print(f"check: no exe at {exe}" if not exe.is_file() else "check: no non-loopback address found; pass --address")
        return 2
    if listening(a.port):
        print(f"check: something already listens on {a.port}; nothing was started")
        return 2

    tmp = Path(tempfile.mkdtemp(prefix="strata-desktop-check-"))
    data = tmp / "d"
    owner_api = "sk-strata-check-" + secrets.token_urlsafe(24)
    keys: list[str] = [owner_api]
    summary: dict = {"exe": str(exe), "exe_sha256": sha256(exe), "exe_size": exe.stat().st_size,
                     "server": f"http://{address}:{a.port}", "bind": "0.0.0.0", "temp": str(tmp)}
    (tmp / "home").mkdir()
    (tmp / "appdata").mkdir()
    cfg = tmp / "check.json"
    cfg.write_text(json.dumps({"api_key": owner_api, "keys_file": "keys.json", "history": {"enabled": False},
                               "model_name": "mock-desktop-check"}), encoding="utf-8")
    server = app = None
    seen_tree: dict[int, str] = {}
    try:
        with open(tmp / "server.out", "wb") as so:
            server = subprocess.Popen([str(server_python()), "-m", "serve.server", "--engine", "mock", "--port",
                                       str(a.port), "--host", "0.0.0.0", "--config", str(cfg)], cwd=ROOT,
                                      stdout=so, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                      creationflags=NO_WINDOW | NEW_GROUP)
        summary["server_pid"] = server.pid
        end = time.monotonic() + 60
        while time.monotonic() < end and server.poll() is None:
            try:
                if req(a.port, "GET", "/health", None, timeout=2)[0] == 200:
                    break
            except OSError:
                pass
            time.sleep(0.5)
        else:
            raise Stop("the mock server did not come up")

        code, body = req(a.port, "POST", "/keys", owner_api, {"action": "create", "name": "owner1", "role": "owner"})
        if code != 200:
            raise Stop(f"creating the owner key: {code}")
        owner = body["key"]
        keys.append(owner)
        code, body = req(a.port, "POST", "/keys", owner, {"action": "create", "name": "ann", "role": "guest"})
        if code != 200:
            raise Stop(f"creating ann: {code}")
        ann = body["key"]
        keys.append(ann)

        # A child the runner owns: an MCP server (serve/mcp_fake_server.py, on the server's python) in the runner's
        # own runner.json, so the selftest's quit step sees a real process of the runner end with Quit.
        (data / "runner").mkdir(parents=True)
        (data / "runner" / "runner.json").write_text(json.dumps({"mcp_servers": {"probe": {
            "command": str(server_python()), "args": [str(ROOT / "serve" / "mcp_fake_server.py")]}}}),
            encoding="utf-8")
        env = {k: v for k, v in os.environ.items() if k != TOKEN_ENV}
        env.update({TOKEN_ENV: ann, "STRATA_HOME": str(tmp / "home"), "APPDATA": str(tmp / "appdata")})
        t0 = time.monotonic()
        app = subprocess.Popen([str(exe), "--selftest", "--server", summary["server"], "--data-dir", str(data)],
                               cwd=tmp, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, creationflags=NO_WINDOW | NEW_GROUP)
        summary["exe_pid"] = app.pid
        result_path = data / "selftest.json"
        approved, runner_seen, t_approved, t_online = None, None, None, None
        while app.poll() is None and time.monotonic() - t0 < EXE_CAP_S:
            for pid, name in tree(app.pid):
                seen_tree.setdefault(pid, name)
            doc = read_result(result_path) or {}
            codes = [s.get("code") for s in doc.get("steps", []) if s.get("step") == "pending"]
            if codes and approved is None:
                code, body = req(a.port, "GET", "/devices/pending", owner)
                row = next((r for r in (body or {}).get("devices", []) if r.get("key") == "ann"
                            and r.get("code") == codes[0]), None)
                if row is not None:
                    code, _ = req(a.port, "POST", "/devices", owner, {"key": "ann", "device": row["device"],
                                                                      "approve": True})
                    t_approved = time.monotonic()
                    approved = {"code": codes[0], "device": row["device"], "label": row.get("label"),
                                "addr": row.get("addr"), "answer": code,
                                "after_s": round(time.monotonic() - t0, 1)}
            code, body = req(a.port, "GET", "/runners", owner)
            for r in (body or {}).get("runners", []):
                if r.get("who") == "ann" and r.get("online"):
                    runner_seen = {k: r.get(k) for k in ("id", "who", "device", "online", "machine", "shell")}
                    if t_online is None:
                        t_online = time.monotonic()
            # between the approval and the runner's first hello that got in: asked every 50 ms, so the wait the
            # user sees after approving is measured to that resolution
            time.sleep(0.05 if t_approved is not None and t_online is None else 0.25)
        if t_approved is not None and t_online is not None:
            summary["approval_to_runner_online_s"] = round(t_online - t_approved, 2)
        summary["exe_seconds"] = round(time.monotonic() - t0, 1)
        summary["exe_exit"] = app.poll()
        summary["approved"] = approved
        summary["runner_seen"] = runner_seen
        summary["exe_tree_seen"] = sorted({name for name in seen_tree.values()})
        summary["exe_tree_seen_count"] = len(seen_tree)

        # what is left, before anything is ended
        time.sleep(2.0)
        summary["left_of_tree"] = [{"pid": p, "name": n} for p, n in alive(set(seen_tree))
                                   if seen_tree.get(p) == n]
        mine = {server.pid} | {p for p, _ in tree(server.pid)}          # the mock server: ended below
        summary["left_by_command_line"] = [r for r in by_command_line(str(tmp)) if r["pid"] not in mine]
        summary["selftest"] = read_result(result_path)
        code, body = req(a.port, "GET", "/keys", owner)
        row = next((k for k in (body or {}).get("keys", []) if k.get("name") == "ann"), {})
        summary["ann_devices"] = [{k: d.get(k) for k in ("device", "label", "status", "addr")}
                                  for d in row.get("devices", [])]
        try:
            device_id = json.loads((data / "runner" / "device.json").read_text(encoding="utf-8"))["id"]
        except (OSError, ValueError, KeyError):
            device_id = None
        summary["device_json_id"] = device_id
        code, body = req(a.port, "GET", "/runners", owner)
        summary["runners_after"] = [{k: r.get(k) for k in ("id", "who", "device", "online")}
                                    for r in (body or {}).get("runners", [])]
        summary["token_bytes_in_profile_files"] = token_in_files(data / "webview", ann)
        log = data / "desktop.log"
        summary["token_in_desktop_log"] = any(ann in f.read_text(encoding="utf-8", errors="replace")
                                              for f in data.glob("desktop.log*"))
        summary["token_in_selftest_json"] = result_path.is_file() and ann in result_path.read_text(encoding="utf-8")
        if a.out:
            out = Path(a.out)
            out.mkdir(parents=True, exist_ok=True)
            for f in (result_path, log):
                if f.is_file():
                    shutil.copy2(f, out / f.name)
    except Stop as e:
        summary["error"] = str(e)
    finally:
        for p in (app, server):
            if p is not None:
                summary.setdefault("ended", []).append({"pid": p.pid, "was_running": p.poll() is None,
                                                        "taskkill": end_tree(p.pid)})
        time.sleep(2.0)
        summary["port_listening_after"] = listening(a.port)
        summary["left_after_end"] = by_command_line(str(tmp))
        if not a.keep:
            shutil.rmtree(tmp, ignore_errors=True)
            summary["temp_removed"] = not tmp.exists()

    s = summary.get("selftest") or {}
    steps = {x.get("step"): x for x in s.get("steps", [])}
    devices = summary.get("ann_devices") or []
    want = {"store", "probe", "runner", "window", "runners", "quit"} | ({"pending"} if summary.get("approved") else set())
    quit_detail = (steps.get("quit") or {}).get("detail") or {}
    checks = {
        "exe_exit_0": summary.get("exe_exit") == 0,
        "selftest_ok": s.get("ok") is True and s.get("exit") == 0,
        "every_step_ok": want <= set(steps) and all(x.get("ok") is True for x in steps.values()),
        # the runner's MCP server was running before Quit and gone after it (the selftest's own check, by PID)
        "runner_child_ended_by_quit": bool(quit_detail.get("children_before"))
                                      and (quit_detail.get("checks") or {}).get("runner_children_ended") is True,
        "pending_then_approved": bool(summary.get("approved")) and summary["approved"]["answer"] == 200,
        "runner_hello_as_ann": (summary.get("runner_seen") or {}).get("who") == "ann"
                               and (summary.get("runner_seen") or {}).get("device") == summary.get("device_json_id"),
        "one_device_row_for_ann": len(devices) == 1 and devices[0]["device"] == summary.get("device_json_id")
                                  and devices[0]["status"] == "approved"
                                  and str(devices[0]["label"]).startswith("Strata Desktop on "),
        "nothing_left_of_exe_tree": summary.get("left_of_tree") == [] and summary.get("left_by_command_line") == [],
        "token_not_in_log_or_result": summary.get("token_in_desktop_log") is False
                                      and summary.get("token_in_selftest_json") is False,
        "port_free": summary.get("port_listening_after") is False,
    }
    summary["checks"] = checks
    summary["passed"] = all(checks.values())
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    for k in keys:
        text = text.replace(k, "***")
    print(text)
    if a.out:
        (Path(a.out) / "summary.json").write_text(text, encoding="utf-8")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
