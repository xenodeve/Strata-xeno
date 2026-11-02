"""Breach: get a COM object served by the desktop shell, a process outside the box. ShellWindows is the classic road from
there to running a program as the user. This stops at getting the object and calls nothing on it."""
import ctypes
import sys
import uuid

CLSCTX_LOCAL_SERVER = 4
SHELL_WINDOWS = uuid.UUID("9BA05972-F6A8-11CF-A442-00A0C90A8F39").bytes_le
IDISPATCH = uuid.UUID("00020400-0000-0000-C000-000000000046").bytes_le
ole = ctypes.windll.ole32
ole.CoInitializeEx(None, 2)
instance = ctypes.c_void_p()
status = ole.CoCreateInstance(SHELL_WINDOWS, None, CLSCTX_LOCAL_SERVER, IDISPATCH, ctypes.byref(instance)) & 0xFFFFFFFF
if status:
    print(f"refused: CoCreateInstance, {status:#010x}")
    sys.exit(10)
print("got the shell's object")
