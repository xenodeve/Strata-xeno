"""Not a breach: the test's helper, run outside the box and never inside one. It moves itself onto the box's private
station and desktop, puts a fake text on that station's clipboard, and reads it back with the same calls clipboard_read.py
uses, which proves the reader's path works. It holds the clipboard's owner window until its stdin closes. Nothing here
reaches the developer's clipboard or display.
"""
import ctypes
import hashlib
import sys
from ctypes import wintypes

CF_UNICODETEXT, GMEM_MOVEABLE, GENERIC_ALL, HWND_MESSAGE = 13, 2, 0x10000000, -3
user = ctypes.WinDLL("user32", use_last_error=True)
kernel = ctypes.WinDLL("kernel32", use_last_error=True)
user.OpenWindowStationW.argtypes = (wintypes.LPCWSTR, wintypes.BOOL, wintypes.DWORD)
user.OpenWindowStationW.restype = wintypes.HANDLE
user.SetProcessWindowStation.argtypes = (wintypes.HANDLE,)
user.OpenDesktopW.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
user.OpenDesktopW.restype = wintypes.HANDLE
user.SetThreadDesktop.argtypes = (wintypes.HANDLE,)
user.CreateWindowExW.argtypes = (wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_int,
                                 ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.HWND, wintypes.HMENU,
                                 wintypes.HINSTANCE, wintypes.LPVOID)
user.CreateWindowExW.restype = wintypes.HWND
user.OpenClipboard.argtypes = (wintypes.HWND,)
user.SetClipboardData.argtypes = (wintypes.UINT, wintypes.HANDLE)
user.SetClipboardData.restype = wintypes.HANDLE
user.GetClipboardData.argtypes = (wintypes.UINT,)
user.GetClipboardData.restype = wintypes.HANDLE
kernel.GlobalAlloc.argtypes = (wintypes.UINT, ctypes.c_size_t)
kernel.GlobalAlloc.restype = wintypes.HANDLE
kernel.GlobalLock.argtypes = (wintypes.HANDLE,)
kernel.GlobalLock.restype = wintypes.LPVOID
kernel.GlobalUnlock.argtypes = (wintypes.HANDLE,)


def need(value, what):
    if not value:
        print(f"helper failed at {what}: error {ctypes.get_last_error()}", flush=True)
        sys.exit(2)
    return value


station_name, desktop_name = sys.argv[1].split("\\")
text = sys.argv[2]
station = need(user.OpenWindowStationW(station_name, False, GENERIC_ALL), "OpenWindowStation")
need(user.SetProcessWindowStation(station), "SetProcessWindowStation")
need(user.SetThreadDesktop(need(user.OpenDesktopW(desktop_name, 0, False, GENERIC_ALL), "OpenDesktop")), "SetThreadDesktop")
window = need(user.CreateWindowExW(0, "STATIC", "fake", 0, 0, 0, 0, 0, HWND_MESSAGE, None, None, None), "CreateWindowEx")

need(user.OpenClipboard(window), "OpenClipboard")
need(user.EmptyClipboard(), "EmptyClipboard")
data = (text + "\0").encode("utf-16-le")
memory = need(kernel.GlobalAlloc(GMEM_MOVEABLE, len(data)), "GlobalAlloc")
ctypes.memmove(kernel.GlobalLock(memory), data, len(data))
kernel.GlobalUnlock(memory)
need(user.SetClipboardData(CF_UNICODETEXT, memory), "SetClipboardData")
user.CloseClipboard()

need(user.OpenClipboard(window), "OpenClipboard again")
held = need(user.GetClipboardData(CF_UNICODETEXT), "GetClipboardData")
back = ctypes.wstring_at(kernel.GlobalLock(held))
kernel.GlobalUnlock(held)
user.CloseClipboard()
print(hashlib.sha256(back.encode("utf-8")).hexdigest(), flush=True)
sys.stdin.read()
