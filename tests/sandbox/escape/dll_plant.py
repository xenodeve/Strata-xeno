"""Breach: plant a DLL where the interpreter will load it. The planted file is a copy of Windows' own version.dll, which
python311.dll imports, so a load is harmless and shows up in the module list.

  beside   write it next to the interpreter, the place the loader looks first
  cwd      put it next to the program and start the interpreter from there
  copy     the control: copy the interpreter into the box with the DLL beside it. This one must load, inside the box, or
           the module list would prove nothing. Code the box copies for itself runs with the box's rights and no more.
"""
import os
import shutil
import subprocess
import sys

PLANT = "version.dll"
mode, check = sys.argv[1], os.path.abspath("dll_check.py")
windows_copy = os.path.join(os.environ["SystemRoot"], "System32", PLANT)
home = os.path.dirname(sys.executable)


def modules(python, cwd, env=None):
    done = subprocess.run([python, check], capture_output=True, text=True, cwd=cwd, env=env, creationflags=0x08000000)
    if done.returncode:
        print(f"checker failed: {done.returncode} {done.stderr[-300:]}")
        sys.exit(3)
    return [line for line in done.stdout.splitlines() if line.lower().endswith("\\" + PLANT)]


if mode == "beside":
    try:
        shutil.copyfile(windows_copy, os.path.join(home, PLANT))
    except OSError as error:
        print(f"refused: {error!r}")
        sys.exit(10)
    print("planted beside the interpreter")
elif mode == "cwd":
    shutil.copyfile(windows_copy, PLANT)
    loaded = modules(sys.executable, os.getcwd())
    print(f"version.dll loaded from: {loaded}")
    sys.exit(0 if any(os.getcwd().lower() in path.lower() for path in loaded) else 10)
elif mode == "copy":
    os.mkdir("copy")
    for name in ("python.exe", "python311.dll", "python3.dll", "vcruntime140.dll", "vcruntime140_1.dll"):
        shutil.copyfile(os.path.join(home, name), os.path.join("copy", name))
    shutil.copyfile(windows_copy, os.path.join("copy", PLANT))
    here = os.path.abspath("copy")
    loaded = modules(os.path.join(here, "python.exe"), here, dict(os.environ, PYTHONHOME=home))
    print(f"version.dll loaded from: {loaded}")
    sys.exit(0 if any(here.lower() in path.lower() for path in loaded) else 3)
