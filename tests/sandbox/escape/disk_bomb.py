"""Breach: fill the disk. Stops by itself at the ceiling given in MiB, then waits so a watcher has time to act.

Modes, one for each way a polled quota can be missed:
  held      one file, one handle kept open: its size in a directory listing goes stale
  files     many closed files
  stream    an alternate data stream: `name:stream` is not in the file's size
  extend    SetEndOfFile, one call that takes the space and writes nothing, so write counters stay at zero. Python's own
            truncate() is no test of this: the C runtime grows a file by writing zeros, a slow writer and nothing more
  deep      files past 260 characters, where a plain path cannot reach them
"""
import ctypes
import msvcrt
import os
import sys
import time
from ctypes import wintypes

folder, ceiling, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
chunk = b"\0" * (1 << 20)
os.makedirs(folder, exist_ok=True)
base = os.path.join(folder, "bomb")
if mode == "extend":
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.SetFilePointerEx.argtypes = (wintypes.HANDLE, ctypes.c_longlong, ctypes.c_void_p, wintypes.DWORD)
    kernel.SetEndOfFile.argtypes = (wintypes.HANDLE,)
    with open(base, "wb") as out:
        handle = msvcrt.get_osfhandle(out.fileno())
        if not (kernel.SetFilePointerEx(handle, ceiling << 20, None, 0) and kernel.SetEndOfFile(handle)):
            print(f"refused: SetEndOfFile, error {ctypes.get_last_error()}", flush=True)
            sys.exit(10)
        print("extended in one call", flush=True)
        time.sleep(4)
elif mode == "deep":
    path = "\\\\?\\" + os.path.abspath(folder)
    for _ in range(20):
        path += "\\" + "d" * 20
        os.mkdir(path)
    with open(path + "\\bomb", "wb") as out:
        for _ in range(ceiling):
            out.write(chunk)
            out.flush()
        time.sleep(4)
elif mode == "files":
    for index in range(ceiling):
        with open(f"{base}{index}", "wb") as out:
            out.write(chunk)
elif mode in ("held", "stream"):
    with open(base + (":hidden" if mode == "stream" else ""), "wb") as out:
        for _ in range(ceiling):
            out.write(chunk)
            out.flush()
        time.sleep(4)
print("survived", flush=True)
time.sleep(4)
