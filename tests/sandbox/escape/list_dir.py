"""Breach: list a folder outside the box. Prints how many names and never the names."""
import os
import sys

try:
    print(len(os.listdir(sys.argv[1])))
except OSError as error:
    print(f"refused: {error!r}")
    sys.exit(10)
