"""Not a breach: what a box can still see. Takes a JSON list of paths and prints, for each, whether it lists and how many
names, as JSON. It reads no file and prints no name. The rest are facts a box can learn without any file."""
import ctypes
import json
import os
import socket
import subprocess
import sys
import winreg

report = {"paths": {}}
for path in json.loads(sys.argv[1]):
    try:
        report["paths"][path] = len(os.listdir(path))
    except OSError as error:
        report["paths"][path] = type(error).__name__


def hklm_readable():
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion") as key:
            return bool(winreg.QueryValueEx(key, "CurrentBuild")[0])
    except OSError as error:
        return type(error).__name__


def own_loopback():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    client = socket.socket()
    client.settimeout(2)
    try:
        client.connect(server.getsockname())
        return True
    except OSError as error:
        return type(error).__name__


def container_storage():
    name = os.environ["USERPROFILE"].split("\\")[-2]
    key = (r"Software\Classes\Local Settings\Software\Microsoft\Windows\CurrentVersion\AppContainer\Storage"
           + "\\" + name + r"\probe")
    try:
        with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, key, 0, winreg.KEY_WRITE) as handle:
            winreg.SetValueEx(handle, "probe", 0, winreg.REG_SZ, "x")
        return True
    except OSError as error:
        return type(error).__name__


def tool(*argv):
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=10, creationflags=0x08000000)
        return f"exit {done.returncode}: {(done.stdout or done.stderr).strip()[:60]}"
    except OSError as error:
        return type(error).__name__


def processes_visible():
    ids, needed = (ctypes.c_ulong * 4096)(), ctypes.c_ulong()
    ctypes.windll.psapi.EnumProcesses(ids, ctypes.sizeof(ids), ctypes.byref(needed))
    return needed.value // 4


size = ctypes.c_ulong(256)
user_name = ctypes.create_unicode_buffer(256)
ctypes.windll.advapi32.GetUserNameW(user_name, ctypes.byref(size))
report.update({"hklm_readable": hklm_readable(), "own_loopback": own_loopback(), "own_registry_key_writable": container_storage(),
               "processes_visible": processes_visible(),
               "user_name_visible": bool(user_name.value), "computer_name_visible": bool(socket.gethostname()),
               "cpu_count": os.cpu_count(), "git": tool(r"C:\Program Files\Git\cmd\git.exe", "--version"),
               "venv": tool(sys.executable, "-m", "venv", "--without-pip", "made-venv"),
               "pip": tool(sys.executable, "-m", "pip", "--version")})
print(json.dumps(report))
