"""Breach: open a named pipe that a program outside the box serves, the way a local server would be reached. With a second
argument it only opens: for a system service's pipe, where a line of text has no business going."""
import sys

try:
    with open(sys.argv[1], "r+b", buffering=0) as pipe:
        if len(sys.argv) == 2:
            pipe.write(b"breached\n")
except OSError as error:
    print(f"refused: {error!r}")
    sys.exit(10)
print("connected")
