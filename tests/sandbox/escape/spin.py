"""Gentle check of the time limit: burn one core for the seconds given, or sleep through them with `sleep`."""
import sys
import time

end = time.monotonic() + float(sys.argv[1])
if len(sys.argv) > 2:
    time.sleep(float(sys.argv[1]))
while time.monotonic() < end:
    pass
print("survived")
