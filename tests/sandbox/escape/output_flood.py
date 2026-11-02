"""Breach: print without end. Stops by itself at the ceiling given in MiB."""
import sys

line = b"x" * 65535 + b"\n"
for _ in range(int(sys.argv[1]) * 16):
    sys.stdout.buffer.write(line)
    sys.stdout.buffer.flush()
print("survived")
