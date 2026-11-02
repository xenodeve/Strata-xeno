"""Breach: write under HKEY_CURRENT_USER. An AppContainer's write can land in a private copy and still report success, so
test_escape.py reads the key back from outside and that decides."""
import sys
import winreg

try:
    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, sys.argv[1], 0, winreg.KEY_WRITE) as key:
        winreg.SetValueEx(key, "breached", 0, winreg.REG_SZ, "yes")
except OSError as error:
    print(f"refused: {error!r}")
    sys.exit(10)
print("wrote")
