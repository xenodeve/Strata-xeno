"""Breach: resolve a name through the system resolver."""
import socket
import sys

try:
    found = socket.getaddrinfo(sys.argv[1], 443)
except OSError as error:
    print(f"refused: {error!r}")
    sys.exit(10)
print(sorted({entry[4][0] for entry in found}))
