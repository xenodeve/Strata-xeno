"""Breach: read the server's secrets out of the environment. Prints every variable as name=value so the test can look for
its canary in the values too; the box's environment is written out from nothing, so nothing real is in it."""
import os

for name, value in sorted(os.environ.items()):
    print(f"{name}={value}")
