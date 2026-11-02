"""serve/sandbox/isolation.py beyond the breaches: what a box can do, that it fails closed, that the host-side code cannot
be turned against the user's own files, and what the other backends do without a machine to run them on."""
import _winapi
import ctypes
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from ctypes import wintypes
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from serve.sandbox import isolation  # noqa: E402
from serve.sandbox.isolation import (BubblewrapBackend, Isolation, IsolationError, MxcBackend, NoneBackend, Policy,  # noqa: E402
                                     THROWAWAY_ENV)

windows = pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
NO_WINDOW = 0x08000000
SAFE = Policy(memory_bytes=768 << 20, seconds=30, processes=12, disk_bytes=64 << 20)


def sddl(path):
    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi.GetNamedSecurityInfoW.argtypes = (wintypes.LPCWSTR, ctypes.c_int, wintypes.DWORD, wintypes.LPVOID, wintypes.LPVOID,
                                             wintypes.LPVOID, wintypes.LPVOID, ctypes.POINTER(wintypes.LPVOID))
    advapi.ConvertSecurityDescriptorToStringSecurityDescriptorW.argtypes = (wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD,
                                                                            ctypes.POINTER(wintypes.LPWSTR), wintypes.LPVOID)
    kernel.LocalFree.argtypes = (wintypes.LPVOID,)
    descriptor, text = wintypes.LPVOID(), wintypes.LPWSTR()
    assert advapi.GetNamedSecurityInfoW(str(path), 1, 4, None, None, None, None, ctypes.byref(descriptor)) == 0
    assert advapi.ConvertSecurityDescriptorToStringSecurityDescriptorW(descriptor, 1, 4, ctypes.byref(text), None)
    out = text.value
    kernel.LocalFree(text)
    kernel.LocalFree(descriptor)
    return out


def text_of(result):
    return result.stdout.decode(errors="replace") + result.stderr.decode(errors="replace")


@pytest.fixture
def outside_folder():
    folder = Path(tempfile.mkdtemp(prefix="strata-outside-"))
    (folder / "keep.txt").write_text("keep\n", encoding="utf-8")
    (folder / "big.bin").write_bytes(b"\0" * (4 << 20))
    yield folder
    shutil.rmtree(folder, ignore_errors=True)


@windows
class TestWhatABoxCanDo:
    def test_stdin_stdout_stderr_and_exit_code(self):
        program = "import sys; print(sys.stdin.read().upper()); print('to stderr', file=sys.stderr); sys.exit(7)"
        with Isolation.start(SAFE) as box:
            result = box.run(["python", "-c", program], stdin="hello", timeout=20)
        assert (result.exit_code, result.stdout.strip(), result.stderr.strip()) == (7, b"HELLO", b"to stderr")
        assert result.killed is None and not result.truncated

    def test_the_library_set_imports_and_computes(self):
        program = ("import sympy, numpy, mpmath, networkx\n"
                   "x = sympy.symbols('x')\n"
                   "print(sympy.integrate(x ** 2, x), numpy.arange(4).sum(), mpmath.mpf(2) ** 3, "
                   "networkx.path_graph(4).number_of_edges())\n")
        with Isolation.start(SAFE) as box:
            result = box.run(["python", "-"], stdin=program, timeout=25)
        assert result.exit_code == 0, text_of(result)
        assert result.stdout.split() == [b"x**3/3", b"6", b"8.0", b"3"]

    def test_pytest_runs_inside(self):
        with Isolation.start(SAFE) as box:
            (box.workdir / "test_probe.py").write_text("def test_ok():\n    assert 1 + 1 == 2\n\n"
                                                       "def test_no():\n    assert 1 + 1 == 3\n", encoding="utf-8")
            result = box.run(["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_probe.py"], timeout=25)
        assert result.exit_code == 1, text_of(result)
        assert b"1 failed, 1 passed" in result.stdout

    def test_unittest_compileall_and_cmd_run_inside(self):
        with Isolation.start(SAFE) as box:
            (box.workdir / "test_u.py").write_text("import unittest\nclass T(unittest.TestCase):\n    def test(self):\n"
                                                   "        self.assertTrue(True)\n", encoding="utf-8")
            assert box.run(["python", "-m", "unittest", "test_u"], timeout=20).exit_code == 0
            assert box.run(["python", "-m", "compileall", "-q", "test_u.py"], timeout=20).exit_code == 0
            shell = box.run(["cmd.exe", "/c", "echo", "hello"], timeout=20)
            assert (shell.exit_code, shell.stdout.strip()) == (0, b"hello")

    def test_a_sibling_module_and_a_new_file_work(self):
        with Isolation.start(SAFE) as box:
            (box.workdir / "helper.py").write_text("VALUE = 41\n", encoding="utf-8")
            (box.workdir / "main.py").write_text("import helper\nopen('out.txt', 'w').write(str(helper.VALUE + 1))\n",
                                                 encoding="utf-8")
            result = box.run(["python", "main.py"], timeout=20)
            assert result.exit_code == 0, text_of(result)
            assert (box.workdir / "out.txt").read_text(encoding="utf-8") == "42"

    def test_temp_is_inside_the_box_for_children_too(self):
        program = ("import os, subprocess, sys, tempfile\n"
                   "print(tempfile.gettempdir())\n"
                   "print(subprocess.run([sys.executable, '-c', 'import tempfile; print(tempfile.gettempdir())'],\n"
                   "                     capture_output=True, text=True, creationflags=0x08000000).stdout.strip())\n")
        with Isolation.start(SAFE) as box:
            result = box.run(["python", "-"], stdin=program, timeout=20)
            lines = result.stdout.decode().split()
            assert len(lines) == 2 and all(line.startswith(str(box.tmpdir)) for line in lines), text_of(result)

    def test_read_write_and_read_only_policy_folders(self):
        scratch = isolation.Scratch()
        read_only = scratch.make(f"ro-{uuid.uuid4().hex[:8]}")
        writable = scratch.make(f"rw-{uuid.uuid4().hex[:8]}")
        (read_only / "in.txt").write_text("input\n", encoding="utf-8")
        program = ("import sys\n"
                   f"print(open(r'{read_only / 'in.txt'}').read().strip())\n"
                   f"open(r'{writable / 'out.txt'}', 'w').write('output')\n"
                   f"try:\n    open(r'{read_only / 'no.txt'}', 'w')\n"
                   "except PermissionError:\n    print('read only holds')\n")
        with Isolation.start(replace(SAFE, readonly=[read_only], readwrite=[writable])) as box:
            result = box.run(["python", "-"], stdin=program, timeout=20)
        assert result.stdout.split() == [b"input", b"read", b"only", b"holds"], text_of(result)
        assert (writable / "out.txt").read_text(encoding="utf-8") == "output"
        assert "S-1-15-2-" not in sddl(read_only).replace("S-1-15-2-1)", "")

    def test_a_run_can_start_in_a_folder_the_policy_gave_and_pytest_collects_there(self):
        parent = isolation.Scratch().make(f"cwd-{uuid.uuid4().hex[:8]}")
        tree = parent / "copy"
        (tree / "sub").mkdir(parents=True)
        (tree / "test_here.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
        with Isolation.start(replace(SAFE, readwrite=[tree])) as box:
            where = box.run(["python", "-c", "import os; print(os.getcwd())"], timeout=20, cwd=tree / "sub")
            tests = box.run(["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_here.py"], timeout=25, cwd=tree)
        assert where.stdout.decode().strip() == str(tree / "sub"), text_of(where)
        assert tests.exit_code == 0 and b"1 passed" in tests.stdout, text_of(tests)

    def test_a_run_cannot_start_in_a_folder_the_policy_did_not_give(self, outside_folder):
        parent = isolation.Scratch().make(f"cwd-{uuid.uuid4().hex[:8]}")
        (parent / "copy").mkdir()
        (parent / "copy-other").mkdir()
        with Isolation.start(replace(SAFE, readwrite=[parent / "copy"])) as box:
            for where in (outside_folder, parent, parent / "copy-other"):
                with pytest.raises(IsolationError, match="not a folder this box was given"):
                    box.run(["python", "-c", "pass"], timeout=20, cwd=where)

    def test_what_lies_beside_a_given_folder_stays_shut(self):
        parent = isolation.Scratch().make(f"beside-{uuid.uuid4().hex[:8]}")
        (parent / "copy").mkdir()
        (parent / "base.pack").write_bytes(b"what a change is measured against")
        program = (f"import os\nfrom pathlib import Path\nparent = Path(r'{parent}')\n"
                   "print(parent.stat().st_size >= 0)\n"
                   "for action in (lambda: os.listdir(parent), lambda: (parent / 'base.pack').read_bytes(), lambda: (parent / 'new.txt').write_text('x')):\n"
                   "    try:\n        action()\n        print('open')\n"
                   "    except PermissionError:\n        print('shut')\n")
        with Isolation.start(replace(SAFE, readwrite=[parent / "copy"])) as box:
            result = box.run(["python", "-"], stdin=program, timeout=20, cwd=parent / "copy")
            during = sddl(parent)
        assert result.stdout.split() == [b"True", b"shut", b"shut", b"shut"], text_of(result)
        assert during != sddl(parent), "the grant on the parent goes with the box"

    def test_stop_ends_what_runs_and_the_box_goes_on(self):
        with Isolation.start(SAFE) as box:
            threading.Timer(0.5, box.stop).start()
            stopped = box.run(["python", "-c", "import time; time.sleep(30)"], timeout=20)
            again = box.run(["python", "-c", "print(5)"], timeout=20)
        assert stopped.killed == "stop"
        assert (again.exit_code, again.stdout.strip(), again.killed) == (0, b"5", None), text_of(again)

    def test_stop_ends_what_a_finished_run_left_behind(self):
        leave = ("import subprocess, sys\n"
                 "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], creationflags=0x08000000)\n")
        with Isolation.start(SAFE) as box:
            assert box.run(["python", "-c", leave], timeout=20).exit_code == 0
            assert box.usage().processes_active > 0
            box.stop()
            assert box.run(["python", "-c", "print(5)"], timeout=20).stdout.strip() == b"5"
            time.sleep(0.05)
            assert box.usage().processes_active == 0

    def test_a_box_can_run_again_after_a_timeout(self):
        with Isolation.start(SAFE) as box:
            assert box.run(["python", "-c", "import time; time.sleep(30)"], timeout=0.4).killed == "timeout"
            first = box.usage().processes_total
            again = box.run(["python", "-c", "print('alive')"], timeout=20)
            assert (again.exit_code, again.stdout.strip()) == (0, b"alive")
            assert box.usage().processes_total > first, "the count must go on across the new job"

    def test_a_kill_does_not_refill_the_cpu_limit(self):
        """Read from the new job itself: the wall clock would end a box on its own and hide a refill."""
        burn = "import time\nend = time.perf_counter() + 0.5\nwhile time.perf_counter() < end: pass\ntime.sleep(30)\n"
        with Isolation.start(replace(SAFE, seconds=20)) as box:
            assert box.run(["python", "-c", burn], timeout=1.5).killed == "timeout"
            spent = box.usage().cpu_seconds
            assert spent >= 0.4
            assert box.run(["python", "-c", "pass"], timeout=10).exit_code == 0
            limits = isolation._ExtendedLimits()
            assert isolation._k32.QueryInformationJobObject(box._job.handle, isolation._JOB_LIMIT_INFO, ctypes.byref(limits),
                                                            ctypes.sizeof(limits), None)
            left = limits.BasicLimitInformation.PerJobUserTimeLimit / 1e7
        assert abs(left - (20 - spent)) < 0.25, (left, spent)

    def test_the_job_ending_itself_on_cpu_is_reported_as_time(self, monkeypatch):
        """Windows ends the job by itself, with nothing here looking on. The job gets 0.5 s against a wall clock of 12, so a
        stop before 12 s can only be the job's."""
        real = isolation._Job._apply
        monkeypatch.setattr(isolation._Job, "_apply", lambda job, policy, cpu: real(job, policy, 0.5))
        with Isolation.start(replace(SAFE, seconds=12)) as box:
            result = box.run(["python", "-c", "while True: pass"], timeout=11.5)
            assert result.exit_code == isolation._STATUS_QUOTA_EXCEEDED, result
            assert result.killed == "time" and result.wall_ms < 11000, result
            again = box.run(["python", "-c", "print('alive')"], timeout=10)
            assert again.stdout.strip() == b"alive", again

    def test_a_box_that_has_used_its_cpu_refuses_the_next_run(self):
        with Isolation.start(SAFE) as box:
            box._job.terminate()
            box._spent["cpu"] = SAFE.seconds
            with pytest.raises(IsolationError, match="out of time"):
                box.run(["python", "-c", "pass"], timeout=5)

    def test_usage_of_a_closed_box_refuses(self):
        box = Isolation.start(SAFE)
        box.kill()
        with pytest.raises(IsolationError, match="closed"):
            box.usage()

    def test_usage_reports_real_numbers(self):
        program = ("import time\nend = time.perf_counter() + 0.25\nwhile time.perf_counter() < end: pass\n"
                   "open('f', 'wb').write(b'x' * 1000000)\n")
        with Isolation.start(SAFE) as box:
            box.run(["python", "-c", program], timeout=20)
            give_up = time.monotonic() + 2
            while box.usage().processes_active and time.monotonic() < give_up:
                time.sleep(0.01)
            used = box.usage()
        assert used.peak_memory_bytes > 1 << 20 and used.processes_total >= 1
        assert used.cpu_seconds >= 0.1, used
        assert 1000000 <= used.disk_bytes < 2000000 and used.written_bytes >= 1000000

    def test_kill_from_another_thread_ends_a_running_program(self):
        """The user pressing stop: run() is still inside when kill() comes."""
        box = Isolation.start(SAFE)
        threading.Timer(0.3, box.kill).start()
        began = time.monotonic()
        result = box.run(["python", "-c", "import time; time.sleep(20)"], timeout=8)
        spent = time.monotonic() - began
        box.kill()
        assert result.killed == "kill", result
        assert spent < 3, spent
        assert not box.dir.exists()

    def test_kill_is_idempotent_and_a_closed_box_refuses(self):
        box = Isolation.start(SAFE)
        folder, profile = box.dir, box._profile.folder
        box.kill()
        box.kill()
        assert not folder.exists() and not profile.exists()
        with pytest.raises(IsolationError, match="closed"):
            box.run(["python", "-c", "pass"])


@windows
class TestFailsClosed:
    def test_a_policy_path_outside_the_scratch_root_is_refused(self, outside_folder):
        before = sddl(outside_folder)
        for policy in (replace(SAFE, readonly=[outside_folder]), replace(SAFE, readwrite=[outside_folder])):
            with pytest.raises(IsolationError, match="outside the scratch root"):
                Isolation.start(policy)
        assert sddl(outside_folder) == before

    def test_a_junction_into_the_scratch_root_does_not_pass_for_inside(self, outside_folder):
        scratch = isolation.Scratch()
        link = outside_folder / "into-scratch"
        _winapi.CreateJunction(str(scratch.make("linked")), str(link))
        with pytest.raises(IsolationError, match="through a link"):
            Isolation.start(replace(SAFE, readonly=[link]))

    def test_a_scratch_folder_with_a_link_inside_is_refused(self, outside_folder):
        scratch = isolation.Scratch()
        folder = scratch.make(f"with-link-{uuid.uuid4().hex[:8]}")
        _winapi.CreateJunction(str(outside_folder), str(folder / "link"))
        try:
            with pytest.raises(IsolationError, match="is a link"):
                Isolation.start(replace(SAFE, readwrite=[folder]))
        finally:
            os.rmdir(folder / "link")

    def test_a_denied_path_open_to_every_container_is_refused(self):
        windows_file = os.path.join(os.environ["SystemRoot"], "System32", "notepad.exe")
        with pytest.raises(IsolationError, match="cannot be denied"):
            Isolation.start(replace(SAFE, denied=[windows_file]))

    def test_a_stopped_firewall_stops_the_box(self, monkeypatch):
        monkeypatch.setattr(isolation, "_service_running", lambda name: name != "MpsSvc")
        with pytest.raises(IsolationError, match="MpsSvc"):
            Isolation.start(SAFE)

    def test_an_unknown_service_counts_as_not_running(self):
        assert isolation._service_running("MpsSvc") and not isolation._service_running("NoSuchServiceHere")

    def test_a_watcher_that_fails_ends_the_box(self, monkeypatch):
        """A box with no watcher has no disk limit and no clock, so it must not go on quietly."""
        with Isolation.start(SAFE) as box:
            def broken():
                raise OSError("the watcher cannot see")
            monkeypatch.setattr(box, "_disk_used", broken)
            result = box.run(["python", "-c", "open('f', 'wb').write(b'x' * (8 << 20)); import time; time.sleep(20)"], timeout=8)
            assert result.killed == "watch", result
            with pytest.raises(IsolationError, match="watcher"):
                box.run(["python", "-c", "pass"], timeout=5)

    def test_a_flood_cannot_hold_the_reader_past_its_room(self, monkeypatch):
        """A writer faster than the reader leaves the pipe never empty, so draining until empty would not return. A stub
        pipe that always has data: a real pipe has moments of emptiness, and they hide this."""
        class NeverEmpty:
            def PeekNamedPipe(self, handle, buffer, size, read, ready, left):
                ready._obj.value = 1 << 16
                return True

            def ReadFile(self, handle, buffer, size, got, overlapped):
                got._obj.value = size
                return True

        monkeypatch.setattr(isolation, "_k32", NeverEmpty())
        sink = bytearray()
        reader = threading.Thread(target=lambda: isolation._drain(None, sink, 1 << 16), daemon=True)
        reader.start()
        reader.join(3)
        returned = not reader.is_alive()
        monkeypatch.undo()
        assert returned, "the drain did not return"
        assert len(sink) <= 1 << 16

    def test_a_half_built_runtime_of_a_dead_server_is_swept(self):
        scratch = isolation.Scratch()
        done = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True,
                              creationflags=NO_WINDOW)
        leftover = scratch.root / "runtime" / f".building-{done.stdout.strip()}-abcd1234"
        leftover.mkdir()
        (leftover / "half.txt").write_text("x", encoding="utf-8")
        scratch.sweep()
        assert not leftover.exists()

    def test_network_is_refused(self):
        with pytest.raises(IsolationError, match="network"):
            Isolation.start(replace(SAFE, network=True))

    def test_a_folder_it_did_not_make_is_not_a_scratch_root(self, outside_folder):
        with pytest.raises(IsolationError, match="not a scratch root"):
            isolation.Scratch(outside_folder)

    def test_a_process_outside_the_job_never_gets_past_the_check(self):
        sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], creationflags=NO_WINDOW)
        try:
            with Isolation.start(SAFE) as box, pytest.raises(IsolationError, match="not in the job"):
                box._confirm(int(sleeper._handle))
        finally:
            sleeper.kill()

    def test_a_plain_token_never_gets_past_the_check(self):
        sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], creationflags=NO_WINDOW)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        try:
            with Isolation.start(SAFE) as box:
                assert kernel.AssignProcessToJobObject(box._job.handle, int(sleeper._handle))
                with pytest.raises(IsolationError, match="not an AppContainer"):
                    box._confirm(int(sleeper._handle))
        finally:
            sleeper.kill()

    def test_the_other_backends_refuse_here(self, monkeypatch):
        monkeypatch.setenv(THROWAWAY_ENV, "1")
        with pytest.raises(IsolationError, match="refused on Windows"):
            Isolation.start(SAFE, backend="none")
        with pytest.raises(IsolationError, match="Linux only"):
            Isolation.start(SAFE, backend="bubblewrap")
        with pytest.raises(IsolationError):
            Isolation.start(SAFE, backend="mxc")
        with pytest.raises(IsolationError, match="no backend named"):
            Isolation.start(SAFE, backend="nothing")

    def test_the_default_is_appcontainer_even_if_wxc_exec_is_there(self, monkeypatch):
        monkeypatch.setattr(MxcBackend, "executable", staticmethod(lambda: r"C:\fake\wxc-exec.exe"))
        with Isolation.start(SAFE) as box:
            assert box.backend == "appcontainer" and box.isolated

    def test_a_bad_policy_is_refused(self):
        for bad in ({"memory_bytes": 0}, {"seconds": -1}, {"processes": 0}, {"disk_bytes": 0}, {"output_bytes": 0},
                    {"cpu_percent": 101}):
            with pytest.raises(IsolationError):
                Policy(**bad)


@windows
class TestTheHostCannotBeTurned:
    """The host-side code runs with the user's full rights over a folder the box controlled."""

    def test_cleanup_does_not_follow_a_junction(self, outside_folder):
        box = Isolation.start(SAFE)
        _winapi.CreateJunction(str(outside_folder), str(box.workdir / "link"))
        box.kill()
        assert (outside_folder / "keep.txt").read_text(encoding="utf-8") == "keep\n"
        assert not box.dir.exists()

    def test_cleanup_does_not_follow_a_junction_a_box_could_nest(self, outside_folder):
        box = Isolation.start(SAFE)
        (box.tmpdir / "a" / "b").mkdir(parents=True)
        _winapi.CreateJunction(str(outside_folder), str(box.tmpdir / "a" / "b" / "link"))
        box.kill()
        assert (outside_folder / "keep.txt").exists() and (outside_folder / "big.bin").exists()

    def test_the_disk_walk_does_not_follow_a_junction(self, outside_folder):
        with Isolation.start(SAFE) as box:
            _winapi.CreateJunction(str(outside_folder), str(box.workdir / "link"))
            assert box.usage().disk_bytes < 1 << 20

    def test_revoking_does_not_spread_through_a_junction(self, outside_folder):
        scratch = isolation.Scratch()
        writable = scratch.make(f"rw-{uuid.uuid4().hex[:8]}")
        before = sddl(outside_folder), sddl(outside_folder / "keep.txt")
        with Isolation.start(replace(SAFE, readwrite=[writable])):
            _winapi.CreateJunction(str(outside_folder), str(writable / "link"))
        assert (sddl(outside_folder), sddl(outside_folder / "keep.txt")) == before
        assert not (writable / "link").exists()

    def test_no_acl_changes_outside_the_scratch_root(self):
        watched = [Path.home(), Path(sys.base_prefix), Path(sys.base_prefix) / "python.exe", ROOT, ROOT / "README.md"]
        before = {path: sddl(path) for path in watched}
        with Isolation.start(SAFE) as box:
            assert box.run(["python", "-c", "print('ran')"], timeout=20).exit_code == 0
        assert {path: sddl(path) for path in watched} == before


class TestOtherBackends:
    def test_report_names_every_backend_with_a_reason(self):
        report = Isolation.report()
        assert set(report) == {"appcontainer", "mxc", "bubblewrap", "none"}
        assert all(ok or why for ok, why in report.values())

    def test_mxc_config_follows_the_published_schema(self, tmp_path):
        policy = Policy(readonly=[r"C:\ro"], readwrite=[r"C:\rw"], denied=[r"C:\ro\.env"], seconds=12)
        path = MxcBackend.write_config(policy, ["python", "-c", "print(1)"], tmp_path / "config.json", cwd=r"C:\rw")
        config = json.loads(path.read_text(encoding="utf-8"))
        assert config["version"] == "1.0.0" and config["containment"] == "processcontainer"
        assert config["process"]["timeout"] == 12000 and config["process"]["inheritDefaultEnv"] is False
        assert config["process"]["commandLine"] == 'python -c print(1)'
        assert config["filesystem"] == {"readonlyPaths": [r"C:\ro"], "readwritePaths": [r"C:\rw"],
                                        "deniedPaths": [r"C:\ro\.env"]}
        assert config["network"] == {"egress": {"default": "deny"}, "ingress": {"default": "deny", "hostLoopback": "deny"}}
        assert config["ui"] == {"disable": True, "clipboard": "none", "injection": False}

    def test_mxc_refuses_network(self):
        with pytest.raises(IsolationError):
            MxcBackend.config(Policy(network=True), ["python"])

    def test_mxc_without_the_tool_refuses(self, monkeypatch):
        monkeypatch.setattr(MxcBackend, "executable", staticmethod(lambda: None))
        with pytest.raises(IsolationError):
            MxcBackend.start(SAFE)

    def test_mxc_with_the_tool_still_refuses_until_it_has_passed_the_suite(self, monkeypatch):
        monkeypatch.setattr(MxcBackend, "executable", staticmethod(lambda: r"C:\fake\wxc-exec.exe"))
        with pytest.raises(IsolationError, match="escape suite"):
            MxcBackend.start(SAFE)

    def test_bwrap_line_shuts_the_network_and_hides_what_is_denied(self, tmp_path):
        denied_file = tmp_path / "secret"
        denied_file.write_text("x", encoding="utf-8")
        box = type("B", (), {"workdir": tmp_path / "work", "tmpdir": tmp_path / "tmp"})()
        policy = Policy(readonly=["/ro"], readwrite=["/rw"], denied=[str(denied_file), str(tmp_path)])
        line = isolation.bwrap_argv(policy, box, ["python3", "-c", "pass"])
        assert "--unshare-all" in line and "--share-net" not in line and "--die-with-parent" in line
        assert line[line.index("--ro-bind", line.index("/ro") - 1):][1] == "/ro"
        assert ["--ro-bind", "/dev/null", str(denied_file)] == line[line.index(str(denied_file)) - 2:line.index(str(denied_file)) + 1]
        assert line[-4:] == ["--", "python3", "-c", "pass"]

    def test_bwrap_refuses_network(self, tmp_path):
        box = type("B", (), {"workdir": tmp_path, "tmpdir": tmp_path})()
        with pytest.raises(IsolationError):
            isolation.bwrap_argv(Policy(network=True), box, ["python3"])

    def test_none_needs_the_flag_on_linux(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.delenv(THROWAWAY_ENV, raising=False)
        assert NoneBackend.available()[0] is False
        with pytest.raises(IsolationError, match=THROWAWAY_ENV):
            NoneBackend.start(SAFE)
        monkeypatch.setenv(THROWAWAY_ENV, "0")
        assert NoneBackend.available()[0] is False
        monkeypatch.setenv(THROWAWAY_ENV, "1")
        assert NoneBackend.available()[0] is True

    def test_bubblewrap_needs_bwrap(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(shutil, "which", lambda name: None)
        assert BubblewrapBackend.available() == (False, "bwrap is not installed")
        with pytest.raises(IsolationError, match="bwrap"):
            BubblewrapBackend.start(SAFE)
