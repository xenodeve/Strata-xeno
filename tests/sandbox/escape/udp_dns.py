"""Breach: ask a public DNS server over raw UDP. A reply is the proof packets went out and back: exit 0 only on a reply."""
import socket
import struct
import sys

question = struct.pack(">HHHHHH", 0x5354, 0x0100, 1, 0, 0, 0) + b"\x07example\x03com\x00" + struct.pack(">HH", 1, 1)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(3)
try:
    sock.sendto(question, (sys.argv[1], 53))
    reply = sock.recv(512)
except OSError as error:
    print(f"refused: {error!r}")
    sys.exit(10)
print(f"reply {len(reply)} bytes")
