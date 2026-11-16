"""Windows DPAPI for Strata Desktop's tokens (#310 part 3): `CryptProtectData` / `CryptUnprotectData` in the current
user's scope, with `CRYPTPROTECT_UI_FORBIDDEN` and the entropy `b"StrataDesktop/1"`.  A blob decrypts only for the same
Windows user (on this PC, or another PC with that user's roaming profile); any other blob raises `DpapiError`.

Imports cleanly on any system; off Windows every call raises `DpapiError("the token store needs Windows")`.
"""
from __future__ import annotations

import ctypes
import os
from ctypes import wintypes

CRYPTPROTECT_UI_FORBIDDEN = 0x1
ENTROPY = b"StrataDesktop/1"


class DpapiError(Exception):
    """A blob that does not decrypt for this user, or a system without DPAPI."""


class _Blob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]


_api = None


def _calls():
    """crypt32's two calls and kernel32's LocalFree with their types, made once (Windows only)."""
    global _api
    if os.name != "nt":
        raise DpapiError("the token store needs Windows")
    if _api is None:
        crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        args = [ctypes.POINTER(_Blob), wintypes.LPCWSTR, ctypes.POINTER(_Blob), ctypes.c_void_p, ctypes.c_void_p,
                wintypes.DWORD, ctypes.POINTER(_Blob)]
        protect = crypt32.CryptProtectData
        protect.argtypes, protect.restype = args, wintypes.BOOL
        unprotect = crypt32.CryptUnprotectData
        unprotect.argtypes = [ctypes.POINTER(_Blob), ctypes.POINTER(wintypes.LPWSTR)] + args[2:]
        unprotect.restype = wintypes.BOOL
        free = kernel32.LocalFree
        free.argtypes, free.restype = [ctypes.c_void_p], ctypes.c_void_p
        _api = (protect, unprotect, free)
    return _api


def _in(data: bytes) -> tuple[_Blob, ctypes.Array]:
    buf = ctypes.create_string_buffer(data, len(data))
    return _Blob(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char))), buf     # buf keeps the bytes alive


def _out(blob: _Blob, free) -> bytes:
    try:
        return ctypes.string_at(blob.pbData, blob.cbData) if blob.cbData else b""
    finally:
        if blob.pbData:
            free(ctypes.cast(blob.pbData, ctypes.c_void_p))


def protect(data: bytes) -> bytes:
    """`data` encrypted for the current Windows user."""
    protect_, _, free = _calls()
    src, _keep = _in(bytes(data))
    ent, _keep2 = _in(ENTROPY)
    out = _Blob()
    if not protect_(ctypes.byref(src), None, ctypes.byref(ent), None, None, CRYPTPROTECT_UI_FORBIDDEN,
                    ctypes.byref(out)):
        raise DpapiError(f"CryptProtectData failed (Windows error {ctypes.get_last_error()})")
    return _out(out, free)


def unprotect(blob: bytes) -> bytes:
    """The bytes `protect` was given; `DpapiError` for a blob made by another user, changed, or not a DPAPI blob."""
    _, unprotect_, free = _calls()
    src, _keep = _in(bytes(blob))
    ent, _keep2 = _in(ENTROPY)
    out = _Blob()
    if not unprotect_(ctypes.byref(src), None, ctypes.byref(ent), None, None, CRYPTPROTECT_UI_FORBIDDEN,
                      ctypes.byref(out)):
        raise DpapiError(f"this token does not decrypt for this Windows user (Windows error "
                         f"{ctypes.get_last_error()})")
    return _out(out, free)
