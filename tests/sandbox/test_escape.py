"""The escape suite of docs/plans/2026-11-02-sandbox-workspace-design.md section 6: one breach per test, from programs of
our own in tests/sandbox/escape/, against fake targets this file makes. It never touches a real secret, and a model's code
never runs here.

How a test earns "blocked":
- The same program runs outside the box first and must succeed, so a wrong path cannot pass for a wall.
- The verdict comes from outside: the bytes of the target, what a listener saw, a key that is not there, a dead pid. A
  program's own exit is only evidence, and exit 10 means the OS refused; any other exit fails the test.
- Bombs run only after the fixture `limits_confirmed` has seen each limit work on a gentle case, and each bomb also stops
  by itself at a ceiling, so a limit that fails costs a bounded amount.

STRATA_ESCAPE_REPORT=<file> writes each result as JSON lines.
"""
import _winapi
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import ctypes
from ctypes import wintypes
from dataclasses import replace
from pathlib import Path

import pytest

if sys.platform != "win32":
    pytest.skip("the AppContainer suite is Windows only", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import winreg  # noqa: E402

from serve.sandbox.isolation import Isolation, Policy, Scratch, build_runtime  # noqa: E402

ESCAPE = Path(__file__).with_name("escape")
NO_WINDOW = 0x08000000
SAFE = Policy(memory_bytes=512 << 20, seconds=25, processes=8, disk_bytes=64 << 20, cpu_percent=25)
TAG = uuid.uuid4().hex[:8]
TARGET = "target-file-with-a-long-name.txt"
PUBLIC_IP = "1.1.1.1"
PUBLIC_NAME = "example.com"
_results = []


def note(name, evidence):
    _results.append({"test": name, "evidence": evidence})


@pytest.fixture(scope="module", autouse=True)
def report():
    yield
    path = os.environ.get("STRATA_ESCAPE_REPORT")
    if path:
        with open(path, "w", encoding="utf-8") as out:
            for row in _results:
                out.write(json.dumps(row) + "\n")


def outside(program, *args, **options):
    options.setdefault("timeout", 30)
    return subprocess.run([sys.executable, str(ESCAPE / program), *map(str, args)], capture_output=True, text=True,
                          creationflags=NO_WINDOW, **options)


def in_box(box, program, *args, timeout=15, also=()):
    for name in (program, *also):
        shutil.copy(ESCAPE / name, box.workdir)
    return box.run(["python", program, *map(str, args)], timeout=timeout)


def inside(program, *args, policy=SAFE, timeout=15, also=()):
    with Isolation.start(policy) as box:
        return in_box(box, program, *args, timeout=timeout, also=also)


def text_of(result):
    return result.stdout.decode(errors="replace") + result.stderr.decode(errors="replace")


def refused(result, why=None):
    text = text_of(result)
    assert result.exit_code == 10, f"exit {result.exit_code:#x}, the program did not reach a refusal:\n{text}"
    assert "refused:" in text
    if why:
        assert why in text, text
    return text.strip()


def sha(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode("utf-8")).hexdigest()


kernel = ctypes.WinDLL("kernel32", use_last_error=True)
kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
kernel.OpenProcess.restype = wintypes.HANDLE
kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
kernel.WaitForSingleObject.restype = wintypes.DWORD
kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
kernel.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
kernel.GetShortPathNameW.argtypes = (wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD)
SYNCHRONIZE, QUERY_LIMITED, TERMINATE = 0x00100000, 0x1000, 0x0001
STATUS_INVALID_HANDLE, STATUS_HEAP_CORRUPTION = 0xC0000008, 0xC0000374


def open_process(pid):
    return kernel.OpenProcess(SYNCHRONIZE | QUERY_LIMITED | TERMINATE, False, pid)


def short_name(path):
    buffer = ctypes.create_unicode_buffer(1024)
    kernel.GetShortPathNameW(str(path), buffer, 1024)
    return buffer.value


@pytest.fixture(scope="module", autouse=True)
def runtime():
    return build_runtime(Scratch())


@pytest.fixture(scope="module")
def fakes():
    """A fake project on each drive, one beside the real ones so it inherits the same ACLs, and a fake home with a fake
    key directly under the user's profile, where a real .ssh would be."""
    on_c = Path(tempfile.mkdtemp(prefix="strata-escape-")) / "fake-project-with-a-long-name"
    on_d = ROOT / "tests" / "sandbox" / f".fake-{TAG}" / "fake-project-with-a-long-name"
    home = Path.home() / f".strata-escape-{TAG}"
    for folder in (on_c, on_d, home / ".ssh"):
        folder.mkdir(parents=True)
    for folder in (on_c, on_d):
        (folder / TARGET).write_text("original\n", encoding="utf-8")
        (folder / ".env").write_text(f"API_KEY=fake-{TAG}\n", encoding="utf-8")
    (home / ".ssh" / "id_rsa").write_text(f"FAKE KEY {TAG}\n", encoding="utf-8")
    yield {"C": on_c, "D": on_d, "home": home}
    shutil.rmtree(on_c.parent, ignore_errors=True)
    shutil.rmtree(on_d.parent, ignore_errors=True)
    shutil.rmtree(home, ignore_errors=True)


BOMB_BLOCK_MIB = 8


def held_mib(result):
    """What memory_bomb.py really held. The job's own peak also counts the block it refused: with 8, 1 and 0.25 MiB blocks
    under a 64 MiB limit the peak read 6.58, 0.79 and 0.21 MiB over, while python's and conhost's private commit stayed
    under 64 MiB (2026-11-02)."""
    return int(re.search(r"at (\d+) MiB", text_of(result)).group(1))


@pytest.fixture(scope="module")
def limits_confirmed():
    """Each limit, seen working on a gentle case, before any bomb may run."""
    with Isolation.start(replace(SAFE, memory_bytes=64 << 20)) as box:
        result = in_box(box, "memory_bomb.py", 96)
        assert result.exit_code == 10, text_of(result)
        assert held_mib(result) < 64
        assert box.usage().peak_memory_bytes < (64 + BOMB_BLOCK_MIB) << 20
    result = inside("spawn_many.py", 5, policy=replace(SAFE, processes=3))
    assert "started 2 refused 3" in text_of(result), text_of(result)
    with Isolation.start(replace(SAFE, seconds=1)) as box:
        result = in_box(box, "spin.py", 30)
        assert result.killed == "time" and result.wall_ms < 4000, result
    with Isolation.start(replace(SAFE, seconds=1)) as box:
        result = in_box(box, "spin.py", 30, "sleep")
        assert result.killed == "time" and result.wall_ms < 4000, result
    with Isolation.start(replace(SAFE, disk_bytes=4 << 20)) as box:
        result = in_box(box, "disk_bomb.py", ".", 16, "files")
        assert result.killed == "disk", result
    with Isolation.start(SAFE) as box:
        result = in_box(box, "spin.py", 30, timeout=0.5)
        assert result.killed == "timeout", result
    return True


@pytest.mark.parametrize("drive", ["C", "D"])
def test_write_outside_by_direct_path(fakes, drive):
    path = fakes[drive] / TARGET
    original = path.read_bytes()
    control = outside("write_file.py", path)
    assert control.returncode == 0 and path.read_bytes() != original
    path.write_bytes(original)
    result = inside("write_file.py", path)
    note(f"write direct {drive}:", refused(result, "PermissionError"))
    assert path.read_bytes() == original


@pytest.mark.parametrize("form", ["extended", "forward", "stream"])
def test_write_outside_by_other_spellings(fakes, form):
    path = fakes["C"] / TARGET
    spelled = {"extended": "\\\\?\\" + str(path), "forward": str(path).replace("\\", "/"), "stream": f"{path}:hidden"}[form]
    original = path.read_bytes()
    result = inside("write_file.py", spelled)
    note(f"write {form} spelling", refused(result, "PermissionError"))
    assert path.read_bytes() == original


@pytest.mark.parametrize("drive", ["C", "D"])
def test_write_outside_by_short_name(fakes, drive):
    path = fakes[drive] / TARGET
    short = short_name(path)
    if "~" not in short:
        note(f"write short name {drive}:", "NOT TESTED: no 8.3 name on this volume")
        pytest.skip(f"no 8.3 name on {drive}:, not tested")
    original = path.read_bytes()
    control = outside("write_file.py", short)
    assert control.returncode == 0 and path.read_bytes() != original
    path.write_bytes(original)
    result = inside("write_file.py", short)
    note(f"write short name {drive}:", refused(result, "PermissionError"))
    assert path.read_bytes() == original


def test_junction_made_in_the_box(fakes):
    folder, target = fakes["C"], fakes["C"] / TARGET
    original = target.read_bytes()
    result = inside("write_junction.py", folder, TARGET)
    assert result.exit_code in (10, 11), text_of(result)
    note("junction made inside", text_of(result).strip())
    assert target.read_bytes() == original


def test_junction_planted_from_outside(fakes):
    """Covers the write through a junction whether or not a box may make one."""
    folder, target = fakes["C"], fakes["C"] / TARGET
    original = target.read_bytes()
    with Isolation.start(SAFE) as box:
        link = box.workdir / "planted"
        _winapi.CreateJunction(str(folder), str(link))
        control = outside("write_file.py", link / TARGET)
        assert control.returncode == 0 and target.read_bytes() != original
        target.write_bytes(original)
        result = in_box(box, "write_file.py", link / TARGET)
        note("write through a junction", refused(result, "PermissionError"))
    assert target.read_bytes() == original


def test_read_denied_file_in_a_folder_the_box_may_read():
    scratch = Scratch()
    folder = scratch.make(f"ro-{uuid.uuid4().hex[:8]}")
    (folder / "ok.txt").write_bytes(b"fine\n")
    (folder / ".env").write_bytes(b"API_KEY=fake\n")
    with Isolation.start(replace(SAFE, readonly=[folder], denied=[folder / ".env"])) as box:
        allowed = in_box(box, "read_file.py", folder / "ok.txt")
        assert allowed.exit_code == 0 and sha(b"fine\n") in text_of(allowed)
        note("read denied file", refused(in_box(box, "read_file.py", folder / ".env"), "PermissionError"))


def test_read_denied_folder_in_a_folder_the_box_may_read():
    scratch = Scratch()
    folder = scratch.make(f"ro-{uuid.uuid4().hex[:8]}")
    (folder / "secrets").mkdir()
    (folder / "secrets" / "key.txt").write_bytes(b"fake\n")
    (folder / "ok.txt").write_bytes(b"fine\n")
    with Isolation.start(replace(SAFE, readonly=[folder], denied=[folder / "secrets"])) as box:
        assert in_box(box, "read_file.py", folder / "ok.txt").exit_code == 0
        note("read inside a denied folder", refused(in_box(box, "read_file.py", folder / "secrets" / "key.txt"), "PermissionError"))
        note("list a denied folder", refused(in_box(box, "list_dir.py", folder / "secrets"), "PermissionError"))


def test_a_denied_file_stays_denied_for_the_next_box_and_a_box_alongside():
    """Giving a denied path back would reopen it to any box that holds a grant on the parent."""
    scratch = Scratch()
    folder = scratch.make(f"ro-{uuid.uuid4().hex[:8]}")
    (folder / ".env").write_bytes(b"API_KEY=fake\n")
    denying, plain = replace(SAFE, readonly=[folder], denied=[folder / ".env"]), replace(SAFE, readonly=[folder])
    alongside = Isolation.start(plain)
    try:
        with Isolation.start(denying) as first:
            refused(in_box(first, "read_file.py", folder / ".env"), "PermissionError")
        note("denied, box alongside", refused(in_box(alongside, "read_file.py", folder / ".env"), "PermissionError"))
    finally:
        alongside.kill()
    with Isolation.start(plain) as later:
        note("denied, next box", refused(in_box(later, "read_file.py", folder / ".env"), "PermissionError"))


def test_read_denied_file_outside_the_scratch_root(fakes):
    secret = fakes["D"] / ".env"
    assert sha(secret.read_bytes()) in outside("read_file.py", secret).stdout
    result = inside("read_file.py", secret, policy=replace(SAFE, denied=[secret]))
    note("read denied file outside scratch", refused(result, "PermissionError"))


def test_read_fake_ssh_key(fakes):
    key = fakes["home"] / ".ssh" / "id_rsa"
    assert sha(key.read_bytes()) in outside("read_file.py", key).stdout
    note("read .ssh key", refused(inside("read_file.py", key), "PermissionError"))
    assert outside("list_dir.py", key.parent).returncode == 0
    note("list .ssh", refused(inside("list_dir.py", key.parent), "PermissionError"))


def test_list_user_profile():
    assert outside("list_dir.py", Path.home()).returncode == 0
    note("list %USERPROFILE%", refused(inside("list_dir.py", Path.home()), "PermissionError"))


def test_known_leak_a_box_can_tell_whether_a_path_exists(fakes):
    """Not a block: this test pins a hole so it stays written down. A box cannot read or list a folder, but a stat answers
    denied for a name that exists and not-found for one that does not, on any path of the machine."""
    program = "import os, sys\ntry:\n    os.stat(sys.argv[1])\nexcept OSError as e:\n    print(type(e).__name__)\n"
    key = fakes["home"] / ".ssh" / "id_rsa"
    with Isolation.start(SAFE) as box:
        here = box.run(["python", "-c", program, str(key)], timeout=15)
        gone = box.run(["python", "-c", program, str(key.with_name("no_such_key"))], timeout=15)
    answers = here.stdout.strip(), gone.stdout.strip()
    note("KNOWN LEAK: path existence", f"existing name: {answers[0].decode()}, missing name: {answers[1].decode()}")
    assert answers == (b"PermissionError", b"FileNotFoundError")


def test_read_the_real_project():
    """Rule I10, read-only and harmless: this file is not a secret."""
    real = ROOT / "README.md"
    assert sha(real.read_bytes()) in outside("read_file.py", real).stdout
    note("read the real project", refused(inside("read_file.py", real), "PermissionError"))


class Tcp:
    def __init__(self, host):
        self.sock = socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((host, 0))
        self.sock.listen(8)
        self.sock.setblocking(False)
        self.port = self.sock.getsockname()[1]

    def accepted(self, wait=0.5):
        end = time.monotonic() + wait
        seen = 0
        while time.monotonic() < end:
            try:
                peer, _ = self.sock.accept()
                peer.close()
                seen += 1
            except BlockingIOError:
                time.sleep(0.02)
        return seen


class Udp:
    def __init__(self, host):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, 0))
        self.sock.setblocking(False)
        self.port = self.sock.getsockname()[1]

    def received(self, wait=0.5):
        end = time.monotonic() + wait
        seen = 0
        while time.monotonic() < end:
            try:
                self.sock.recv(64)
                seen += 1
            except (BlockingIOError, ConnectionResetError):
                time.sleep(0.02)
        return seen


def lan_address():
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect((PUBLIC_IP, 53))
        return probe.getsockname()[0]
    except OSError:
        return None
    finally:
        probe.close()


@pytest.mark.parametrize("host", ["127.0.0.1", "::1"])
def test_tcp_to_a_loopback_port_the_test_opened(host):
    listener = Tcp(host)
    assert outside("tcp_connect.py", host, listener.port).returncode == 0
    assert listener.accepted() == 1
    result = inside("tcp_connect.py", host, listener.port)
    note(f"TCP loopback {host}", refused(result))
    assert listener.accepted() == 0


def test_tcp_to_this_machines_lan_address():
    host = lan_address()
    if host is None:
        pytest.skip("no LAN address, not tested")
    listener = Tcp(host)
    assert outside("tcp_connect.py", host, listener.port).returncode == 0 and listener.accepted() == 1
    result = inside("tcp_connect.py", host, listener.port)
    note("TCP to the LAN address", refused(result))
    assert listener.accepted() == 0


def test_tcp_out_to_the_internet():
    if outside("tcp_connect.py", PUBLIC_IP, 443).returncode:
        pytest.skip("no internet from outside the box, not tested")
    note("TCP to the internet", refused(inside("tcp_connect.py", PUBLIC_IP, 443)))


@pytest.mark.parametrize("where", ["loopback", "lan"])
def test_udp_to_a_port_the_test_opened(where):
    host = "127.0.0.1" if where == "loopback" else lan_address()
    if host is None:
        pytest.skip("no LAN address, not tested")
    listener = Udp(host)
    assert outside("udp_send.py", host, listener.port).returncode == 0
    assert listener.received() == 3
    result = inside("udp_send.py", host, listener.port)
    seen = listener.received()
    note(f"UDP to {where}", f"sendto exit {result.exit_code}: {text_of(result).strip()}; datagrams received: {seen}")
    assert seen == 0


def test_udp_out_to_the_internet():
    if outside("udp_dns.py", PUBLIC_IP).returncode:
        pytest.skip("no UDP to the internet from outside the box, not tested")
    note("UDP to the internet", refused(inside("udp_dns.py", PUBLIC_IP)))


def test_dns_through_the_system_resolver():
    if outside("dns_lookup.py", PUBLIC_NAME).returncode:
        pytest.skip("no DNS from outside the box, not tested")
    note("DNS lookup", refused(inside("dns_lookup.py", PUBLIC_NAME)))


def test_a_box_name_does_not_reach_the_resolver():
    """A lookup that fails in the box could still leave the machine, with data in the name. Windows caches a failed name,
    so `ipconfig /displaydns` shows whether the resolver ever saw it. The control shows that it would."""
    def resolver_saw(name):
        return name in subprocess.run(["ipconfig", "/displaydns"], capture_output=True, text=True, errors="replace",
                                      creationflags=NO_WINDOW).stdout.lower()
    seen_name = f"strata-control-{uuid.uuid4().hex[:10]}.example.com"
    secret_name = f"strata-escape-{uuid.uuid4().hex[:10]}.example.com"
    outside("dns_lookup.py", seen_name)
    if not resolver_saw(seen_name):
        note("DNS name leaves through the resolver", "NOT TESTED: Windows did not cache the failed control name")
        pytest.skip("the control name was not cached, not tested")
    refused(inside("dns_lookup.py", secret_name))
    note("DNS name leaves through the resolver", "the resolver's cache never held the box's name")
    assert not resolver_saw(secret_name)


def test_icmp_out_to_the_internet():
    if outside("icmp_ping.py", PUBLIC_IP).returncode:
        pytest.skip("no ICMP reply from outside the box, not tested")
    note("ICMP echo", refused(inside("icmp_ping.py", PUBLIC_IP)))


def test_named_pipe_of_a_program_outside():
    name = f"\\\\.\\pipe\\strata-escape-{uuid.uuid4().hex[:8]}"
    served = [_winapi.CreateNamedPipe(name, _winapi.PIPE_ACCESS_DUPLEX, _winapi.PIPE_WAIT, 4, 4096, 4096, 0, _winapi.NULL)
              for _ in range(2)]
    try:
        assert outside("pipe_open.py", name).returncode == 0
        note("named pipe", refused(inside("pipe_open.py", name)))
    finally:
        for handle in served:
            _winapi.CloseHandle(handle)


def test_reach_a_process_outside():
    def sleeper():
        return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], creationflags=NO_WINDOW)
    control = sleeper()
    assert outside("process_open.py", control.pid).returncode == 0
    assert control.wait(5) == 9
    target = sleeper()
    try:
        note("reach a process outside", refused(inside("process_open.py", target.pid)))
        assert target.poll() is None
    finally:
        target.kill()


def test_detached_child_ends_with_the_box():
    control = outside("detached_child.py", 8)
    pid = int(re.search(r"pid (\d+)", control.stdout).group(1))
    handle = open_process(pid)
    assert handle and kernel.WaitForSingleObject(handle, 0) != 0, "the control's child must outlive its parent"
    kernel.TerminateProcess(handle, 0)
    kernel.CloseHandle(handle)

    box = Isolation.start(SAFE)
    try:
        result = in_box(box, "detached_child.py", 20)
        text = text_of(result)
        assert "breakaway refused" in text, text
        pid = int(re.search(r"stayed pid (\d+)", text).group(1))
        handle = open_process(pid)
        assert handle and kernel.WaitForSingleObject(handle, 0) != 0, "the child must be alive before the kill"
    finally:
        box.kill()
    try:
        assert kernel.WaitForSingleObject(handle, 3000) == 0, "the child outlived the box"
    finally:
        kernel.TerminateProcess(handle, 0)
        kernel.CloseHandle(handle)
    note("detached child", "breakaway refused by the job; the child stayed in it and died with the box")


def test_box_ends_when_the_host_dies():
    done = outside("host_dies.py", ROOT)
    text = done.stdout
    assert "stayed pid" in text, text + done.stderr
    pid = int(re.search(r"stayed pid (\d+)", text).group(1))
    name = re.search(r"box (\S+)", text).group(1)
    time.sleep(0.5)
    handle = open_process(pid)
    if handle:
        try:
            assert kernel.WaitForSingleObject(handle, 3000) == 0, "the box outlived the host"
        finally:
            kernel.CloseHandle(handle)
    scratch = Scratch()
    assert (scratch.root / "boxes" / name).exists(), "a crash leaves the folder for the sweep"
    scratch.sweep()
    assert not (scratch.root / "boxes" / name).exists()
    note("host crash", "kill-on-close ended the box's child; the next sweep removed its folder")


def test_task_scheduler_from_the_box():
    control = outside("task_service.py")
    assert control.returncode == 0, control.stdout + control.stderr
    result = inside("task_service.py")
    assert result.exit_code in (10, STATUS_HEAP_CORRUPTION), f"{result.exit_code:#x} {text_of(result)}"
    note("Task Scheduler client", f"no connection, exit {result.exit_code:#x}: {text_of(result).strip()}; the client failed "
                                  "inside the box, which does not show the service refused")


def test_task_scheduler_pipe_from_the_box():
    pipe = "\\\\.\\pipe\\atsvc"
    if outside("pipe_open.py", pipe, "open only").returncode:
        note("Task Scheduler pipe", "NOT TESTED: the pipe does not open from outside the box either")
        pytest.skip("atsvc does not open from outside the box, not tested")
    note("Task Scheduler pipe", refused(inside("pipe_open.py", pipe, "open only"), "PermissionError"))


def test_com_server_of_the_desktop_shell():
    if outside("com_shell.py").returncode:
        pytest.skip("no shell COM object from outside the box, not tested")
    note("COM server in the shell", refused(inside("com_shell.py")))


def test_registry_write_to_hkcu():
    key = f"Software\\StrataEscape-{TAG}-write"
    assert outside("registry_write.py", key).returncode == 0
    winreg.DeleteKey(winreg.HKEY_CURRENT_USER, key)
    result = inside("registry_write.py", key)
    try:
        with pytest.raises(FileNotFoundError):
            winreg.OpenKey(winreg.HKEY_CURRENT_USER, key)
    finally:
        try:
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, key)
        except OSError:
            pass
    note("registry write HKCU", refused(result, "PermissionError"))


def test_registry_read_of_hkcu():
    key = f"Software\\StrataEscape-{TAG}-read"
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key) as handle:
        winreg.SetValueEx(handle, "fake", 0, winreg.REG_SZ, f"fake-{TAG}")
    try:
        assert f"fake-{TAG}" in outside("registry_read.py", key, "fake").stdout
        note("registry read HKCU", refused(inside("registry_read.py", key, "fake"), "PermissionError"))
    finally:
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, key)


def test_clipboard_read():
    """The fake text sits on the clipboard of the box's own station: the developer's clipboard is never written or read."""
    text = f"fake-clipboard-{TAG}"
    with Isolation.start(SAFE) as box:
        helper = subprocess.Popen([sys.executable, str(ESCAPE / "clipboard_fake.py"), box._desktop_name, text],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, creationflags=NO_WINDOW)
        try:
            assert helper.stdout.readline().strip() == sha(text), "the fake must be readable from outside the box"
            result = in_box(box, "clipboard_read.py")
            note("clipboard read", refused(result))
            assert sha(text) not in text_of(result)
        finally:
            helper.stdin.close()
            helper.wait(5)


def test_the_box_cannot_open_the_developers_desktop():
    assert outside("winsta0.py").returncode == 0
    result = inside("winsta0.py")
    text = refused(result)
    note("open WinSta0", text)
    assert "on WinSta0" not in text


def test_dll_beside_the_interpreter():
    note("DLL beside the interpreter", refused(inside("dll_plant.py", "beside", also=["dll_check.py"]), "PermissionError"))


def test_dll_next_to_the_program():
    result = inside("dll_plant.py", "cwd", also=["dll_check.py"])
    text = refused_or_module_list(result)
    note("DLL next to the program", text)
    assert "system32" in text.lower()


def refused_or_module_list(result):
    text = text_of(result)
    assert result.exit_code == 10, f"exit {result.exit_code:#x}: the planted DLL loaded or the check failed\n{text}"
    return text.strip()


def test_the_dll_check_can_see_a_planted_dll():
    """The control of the two tests above, and a limit worth stating: an interpreter the box copies for itself does load
    the DLL beside it. That code has the box's rights and no more, so it is not a breach, and not something a wall stops."""
    result = inside("dll_plant.py", "copy", also=["dll_check.py"])
    assert result.exit_code == 0, text_of(result)
    note("DLL control", text_of(result).strip())


def test_environment_of_the_server_stays_out(monkeypatch):
    canary = f"fake-secret-{TAG}"
    monkeypatch.setenv("STRATA_ESCAPE_CANARY", canary)
    assert canary in outside("env_names.py").stdout
    result = inside("env_names.py")
    assert result.exit_code == 0 and canary not in text_of(result)
    note("server environment", f"{len(text_of(result).splitlines())} variables, canary absent")


def test_inheritable_handle_stays_out(fakes):
    secret = fakes["C"] / ".env"
    descriptor = os.open(secret, os.O_RDONLY | os.O_BINARY)
    import msvcrt
    handle = msvcrt.get_osfhandle(descriptor)
    os.set_handle_inheritable(handle, True)
    try:
        control = subprocess.run([sys.executable, str(ESCAPE / "handle_leak.py"), str(handle)], capture_output=True, text=True,
                                 close_fds=False, creationflags=NO_WINDOW, timeout=30)
        assert sha(secret.read_bytes()) in control.stdout, control.stdout + control.stderr
        os.lseek(descriptor, 0, os.SEEK_SET)
        result = inside("handle_leak.py", handle)
        assert sha(secret.read_bytes()) not in text_of(result)
        assert result.exit_code == STATUS_INVALID_HANDLE, f"{result.exit_code:#x} {text_of(result)}"
        note("inherited handle", "no such handle in the box: the kernel ended the program with STATUS_INVALID_HANDLE")
    finally:
        os.close(descriptor)


def test_a_box_cannot_reach_another_box():
    with Isolation.start(SAFE) as first, Isolation.start(SAFE) as second:
        mine = first.workdir / "mine.txt"
        assert in_box(first, "write_file.py", mine).exit_code == 0
        assert mine.read_text(encoding="utf-8") == "breached\n"
        note("read another box", refused(in_box(second, "read_file.py", mine), "PermissionError"))
        note("write another box", refused(in_box(second, "write_file.py", mine), "PermissionError"))
        assert mine.read_text(encoding="utf-8") == "breached\n"


def test_the_runtime_is_read_only_for_a_box(runtime):
    target = runtime / "Lib" / "os.py"
    original = target.read_bytes()
    note("write the runtime", refused(inside("write_file.py", target), "PermissionError"))
    assert target.read_bytes() == original


def test_the_scratch_root_is_not_listable_from_a_box():
    root = Scratch().root
    note("list the scratch root", refused(inside("list_dir.py", root / "boxes"), "PermissionError"))


def test_fork_bomb(limits_confirmed):
    samples = []
    stop = threading.Event()
    limit = 6
    policy = replace(SAFE, processes=limit, cpu_percent=10, memory_bytes=384 << 20)
    with Isolation.start(policy) as box:
        def sample():
            while not stop.wait(0.02):
                samples.append(box.usage().processes_active)
        watcher = threading.Thread(target=sample, daemon=True)
        watcher.start()
        try:
            result = in_box(box, "fork_bomb.py", 5, timeout=20)
        finally:
            stop.set()
            watcher.join()
        total = box.usage().processes_total
    text = text_of(result)
    refusals = text.count("refused:")
    note("fork bomb", f"{refusals} spawns refused; at most {max(samples)} active counting one console host per program, limit "
                      f"{limit} programs; {total} ever started; 63 programs would run with no limit")
    assert refusals > 0
    assert max(samples) <= limit * 2


def test_memory_bomb(limits_confirmed):
    limit = 128 << 20
    with Isolation.start(replace(SAFE, memory_bytes=limit, cpu_percent=10)) as box:
        result = in_box(box, "memory_bomb.py", 2048)
        peak = box.usage().peak_memory_bytes
    text = refused(result, "MemoryError")
    note("memory bomb", f"{text}; job peak {peak >> 20} MiB counting the refused block, limit {limit >> 20} MiB, "
                        "the bomb asked for 2048 MiB")
    assert held_mib(result) < limit >> 20
    assert peak < limit + (BOMB_BLOCK_MIB << 20)


def test_a_box_cannot_connect_to_another_box():
    def serve_in(box, *mode):
        shutil.copy(ESCAPE / "tcp_connect.py", box.workdir)
        seen = {}
        worker = threading.Thread(target=lambda: seen.update(result=in_box(box, "listen_once.py", 2.5, *mode)), daemon=True)
        worker.start()
        return worker, seen

    with Isolation.start(SAFE) as server, Isolation.start(SAFE) as other:
        worker, seen = serve_in(server)
        give_up = time.monotonic() + 8
        while not (server.workdir / "port.txt").exists() and time.monotonic() < give_up:
            time.sleep(0.01)
        port = (server.workdir / "port.txt").read_text(encoding="utf-8")
        attempt = in_box(other, "tcp_connect.py", "127.0.0.1", port)
        worker.join(10)
        assert b"nobody came" in seen["result"].stdout, seen["result"]
        note("TCP to another box", refused(attempt))
        (server.workdir / "port.txt").unlink()
        worker, seen = serve_in(server, "pair")
        worker.join(10)
        assert b"accepted" in seen["result"].stdout, seen["result"]
        note("TCP inside one box (control)", "accepted: loopback between programs of one box works")


def test_files_a_box_leaves_do_not_outlive_it():
    box = Isolation.start(SAFE)
    result = in_box(box, "resist_cleanup.py")
    text = text_of(result)
    worked = text[text.index("worked: ") + 8:].strip()
    box.kill()
    note("files that resist cleanup", f"tried and worked: {worked}; {text[:text.index('worked: ')].strip()}")
    assert "read-only folder" in worked and "past 260 characters" in worked and "reserved name" in worked
    assert not box.dir.exists(), f"left behind: {sorted(os.listdir(box.dir))}"


@pytest.mark.parametrize("mode", ["held", "files", "stream", "extend", "deep"])
def test_disk_bomb(limits_confirmed, mode):
    """The watcher stops the box, and that is all it can promise. The limit is a poll: a writer gets past it by what it
    writes between two polls, and `extend` takes all it asked for in one call before any poll can see it."""
    limit, ceiling = 32 << 20, 256
    with Isolation.start(replace(SAFE, disk_bytes=limit, seconds=15)) as box:
        result = in_box(box, "disk_bomb.py", ".", ceiling, mode, timeout=12)
        used = box.usage().disk_bytes
    note(f"disk bomb ({mode})", f"killed by {result.killed}, {used >> 20} MiB on disk at the end, limit {limit >> 20} MiB, "
                                f"the bomb asked for {ceiling} MiB")
    assert result.killed == "disk", (result.exit_code, text_of(result))
    assert "survived" not in text_of(result)


def test_disk_bomb_in_the_profile_folder(limits_confirmed):
    """The one writable place outside the scratch root: CreateAppContainerProfile makes it and the box owns it."""
    limit, ceiling = 16 << 20, 128
    with Isolation.start(replace(SAFE, disk_bytes=limit, seconds=15)) as box:
        folder = box._profile.folder
        result = in_box(box, "disk_bomb.py", folder / "Temp", ceiling, "held", timeout=12)
    note("disk bomb in the AC folder", f"killed by {result.killed}")
    assert result.killed == "disk", (result.exit_code, text_of(result))
    assert not folder.exists(), "the profile's folder must go with the box"


def test_output_flood(limits_confirmed):
    cap = 1 << 20
    with Isolation.start(replace(SAFE, output_bytes=cap, cpu_percent=10)) as box:
        result = in_box(box, "output_flood.py", 512)
    size = len(result.stdout) + len(result.stderr)
    note("output flood", f"killed by {result.killed}, {size} bytes kept of a cap of {cap}, the flood offered 512 MiB")
    assert result.killed == "output" and result.truncated
    assert size <= cap
    assert b"survived" not in result.stdout
