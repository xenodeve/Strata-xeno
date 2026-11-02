"""Breach: send an ICMP echo out of the box. Exit 0 only on a reply."""
import ctypes
import socket
import struct
import sys
from ctypes import wintypes

icmp = ctypes.WinDLL("iphlpapi", use_last_error=True)
icmp.IcmpCreateFile.restype = wintypes.HANDLE
icmp.IcmpSendEcho.argtypes = (wintypes.HANDLE, wintypes.DWORD, wintypes.LPVOID, wintypes.WORD, wintypes.LPVOID,
                              wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD)
icmp.IcmpCloseHandle.argtypes = (wintypes.HANDLE,)
handle = icmp.IcmpCreateFile()
if handle in (None, wintypes.HANDLE(-1).value):
    print(f"refused: no ICMP handle, {ctypes.get_last_error()}")
    sys.exit(10)
address = struct.unpack("<I", socket.inet_aton(sys.argv[1]))[0]
reply = ctypes.create_string_buffer(256)
payload = ctypes.create_string_buffer(b"breached")
answers = icmp.IcmpSendEcho(handle, address, payload, 8, None, reply, 256, 2000)
icmp.IcmpCloseHandle(handle)
if not answers:
    print(f"refused: no echo reply, {ctypes.get_last_error()}")
    sys.exit(10)
print("echo reply")
