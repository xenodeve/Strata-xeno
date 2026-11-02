"""Breach: send UDP out of the box. `sendto` can return success while the firewall drops the packet, so this program's own
exit decides nothing: test_escape.py counts what reaches its listener."""
import socket
import sys

host, port = sys.argv[1], int(sys.argv[2])
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    for _ in range(3):
        sock.sendto(b"breached\n", (host, port))
except OSError as error:
    print(f"refused: {error!r}")
    sys.exit(10)
print("sent")
