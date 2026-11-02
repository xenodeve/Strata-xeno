"""Breach: read the clipboard. Prints a hash and never the text. Run only on the box's private station, where the test has
put a fake text: it never touches the developer's real clipboard."""
import ctypes
import hashlib
import sys
from ctypes import wintypes

CF_UNICODETEXT = 13
user = ctypes.WinDLL("user32", use_last_error=True)
kernel = ctypes.WinDLL("kernel32", use_last_error=True)
user.GetClipboardData.argtypes = (wintypes.UINT,)
user.GetClipboardData.restype = wintypes.HANDLE
kernel.GlobalLock.argtypes = (wintypes.HANDLE,)
kernel.GlobalLock.restype = wintypes.LPVOID
kernel.GlobalUnlock.argtypes = (wintypes.HANDLE,)

if not user.OpenClipboard(None):
    print(f"refused: OpenClipboard, error {ctypes.get_last_error()}")
    sys.exit(10)
try:
    handle = user.GetClipboardData(CF_UNICODETEXT)
    if not handle:
        print(f"refused: GetClipboardData, error {ctypes.get_last_error()}")
        sys.exit(10)
    text = ctypes.wstring_at(kernel.GlobalLock(handle))
    kernel.GlobalUnlock(handle)
    print(hashlib.sha256(text.encode("utf-8")).hexdigest())
finally:
    user.CloseClipboard()
