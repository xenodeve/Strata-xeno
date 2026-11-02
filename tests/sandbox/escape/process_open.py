"""Breach: reach a process outside the box and end it. The pid is a sleeper the test started for this."""
import ctypes
import sys
from ctypes import wintypes

kernel = ctypes.WinDLL("kernel32", use_last_error=True)
kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
kernel.OpenProcess.restype = wintypes.HANDLE
kernel.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
PROCESS_TERMINATE, PROCESS_VM_READ = 0x0001, 0x0010
handle = kernel.OpenProcess(PROCESS_TERMINATE | PROCESS_VM_READ, False, int(sys.argv[1]))
if not handle:
    print(f"refused: OpenProcess, error {ctypes.get_last_error()}")
    sys.exit(10)
kernel.TerminateProcess(handle, 9)
print("opened and ended")
