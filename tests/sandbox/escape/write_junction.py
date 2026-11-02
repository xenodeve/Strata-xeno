"""Breach: make a junction inside the box that points outside it, then write through it. Exits 11 if the junction itself
was refused, which leaves the write untested: test_escape.py plants one from outside to cover that."""
import _winapi
import os
import sys

target, name = sys.argv[1], sys.argv[2]
link = os.path.abspath("link")
try:
    _winapi.CreateJunction(target, link)
except OSError as error:
    print(f"no junction: {error!r}")
    sys.exit(11)
print("junction made")
try:
    with open(os.path.join(link, name), "a", encoding="utf-8") as out:
        out.write("breached\n")
except OSError as error:
    print(f"refused: {error!r}")
    sys.exit(10)
print("wrote")
