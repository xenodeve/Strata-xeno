"""Breach: reach the developer's desktop session. Opening WinSta0 is the door to their real clipboard and screen, so this
stops at the open and reads nothing through it. Prints where the program itself is."""
import ctypes
import sys
from ctypes import wintypes

user = ctypes.WinDLL("user32", use_last_error=True)
kernel = ctypes.WinDLL("kernel32", use_last_error=True)
user.GetProcessWindowStation.restype = wintypes.HANDLE
user.GetThreadDesktop.restype = wintypes.HANDLE
user.GetThreadDesktop.argtypes = (wintypes.DWORD,)
user.GetUserObjectInformationW.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD,
                                           ctypes.POINTER(wintypes.DWORD))
user.OpenWindowStationW.argtypes = (wintypes.LPCWSTR, wintypes.BOOL, wintypes.DWORD)
user.OpenWindowStationW.restype = wintypes.HANDLE


def name_of(handle):
    name, got = ctypes.create_unicode_buffer(256), wintypes.DWORD()
    user.GetUserObjectInformationW(handle, 2, name, ctypes.sizeof(name), ctypes.byref(got))
    return name.value


print(f"on {name_of(user.GetProcessWindowStation())}\\{name_of(user.GetThreadDesktop(kernel.GetCurrentThreadId()))}")
WINSTA_ACCESSCLIPBOARD, WINSTA_READSCREEN = 0x0004, 0x0200
door = user.OpenWindowStationW("WinSta0", False, WINSTA_ACCESSCLIPBOARD | WINSTA_READSCREEN)
if not door:
    print(f"refused: OpenWindowStation, error {ctypes.get_last_error()}")
    sys.exit(10)
print("opened WinSta0")
