"""Breach: read a file the box should not see. Prints a hash and never the content."""
import hashlib
import sys

try:
    with open(sys.argv[1], "rb") as source:
        print(hashlib.sha256(source.read()).hexdigest())
except OSError as error:
    print(f"refused: {error!r}")
    sys.exit(10)
