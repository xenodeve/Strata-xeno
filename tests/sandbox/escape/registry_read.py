"""Breach: read a value under HKEY_CURRENT_USER. The value is one the test made."""
import sys
import winreg

try:
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, sys.argv[1]) as key:
        print(winreg.QueryValueEx(key, sys.argv[2])[0])
except OSError as error:
    print(f"refused: {error!r}")
    sys.exit(10)
