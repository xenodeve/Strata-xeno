"""Not a breach by itself: lists every module this process loaded, with its path. dll_plant.py starts it."""
import ctypes
import sqlite3  # noqa: F401
import ssl  # noqa: F401
import sys
from ctypes import wintypes

psapi = ctypes.WinDLL("psapi", use_last_error=True)
kernel = ctypes.WinDLL("kernel32", use_last_error=True)
kernel.GetCurrentProcess.restype = wintypes.HANDLE
kernel.GetModuleFileNameW.argtypes = (wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD)
psapi.EnumProcessModules.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.HMODULE), wintypes.DWORD,
                                     ctypes.POINTER(wintypes.DWORD))
modules, needed = (wintypes.HMODULE * 1024)(), wintypes.DWORD()
if not psapi.EnumProcessModules(kernel.GetCurrentProcess(), modules, ctypes.sizeof(modules), ctypes.byref(needed)):
    sys.exit(2)
for index in range(needed.value // ctypes.sizeof(wintypes.HMODULE)):
    path = ctypes.create_unicode_buffer(1024)
    kernel.GetModuleFileNameW(modules[index], path, 1024)
    print(path.value)
