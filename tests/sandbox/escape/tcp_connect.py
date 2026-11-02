"""Breach: open a TCP connection out of the box and send a line."""
import socket
import sys

host, port = sys.argv[1], int(sys.argv[2])
sock = socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(3)
try:
    sock.connect((host, port))
    sock.sendall(b"breached\n")
except OSError as error:
    print(f"refused: {error!r}")
    sys.exit(10)
print("connected")
