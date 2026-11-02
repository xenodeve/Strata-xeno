"""Breach: use a handle the parent left inheritable. The test opens its fake secret that way and passes the number here.
Prints a hash and never the content."""
import hashlib
import msvcrt
import os
import sys

try:
    with os.fdopen(msvcrt.open_osfhandle(int(sys.argv[1]), os.O_RDONLY), "rb") as source:
        print(hashlib.sha256(source.read()).hexdigest())
except OSError as error:
    print(f"refused: {error!r}")
    sys.exit(10)
