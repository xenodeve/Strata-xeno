"""Breach: change a file outside the box through the path given (plain, short name, or through a junction).

Every program here exits 0 if the breach worked and 10 if the OS refused it. Any other exit means the program itself
broke, which test_escape.py treats as a failure, never as blocked.
"""
import sys

try:
    with open(sys.argv[1], "a", encoding="utf-8") as target:
        target.write("breached\n")
except OSError as error:
    print(f"refused: {error!r}")
    sys.exit(10)
print("wrote")
