"""Breach: leave files the host cannot delete, so they outlive the box. Every trick a box can reach by itself: read-only,
hidden and system marks, a reserved device name, a trailing dot, a path past 260 characters, and an ACL that shuts the
user out. Prints what worked, so the test knows each trick was really tried."""
import os
import stat
import subprocess
import sys

here = "\\\\?\\" + os.getcwd()
done = []


def attempt(name, work):
    try:
        work()
        done.append(name)
    except OSError as error:
        print(f"{name} failed: {error!r}", flush=True)


def write(path):
    with open(path, "wb") as out:
        out.write(b"x")


def read_only_folder():
    os.mkdir(here + "\\locked")
    write(here + "\\locked\\inside.txt")
    os.chmod(here + "\\locked\\inside.txt", stat.S_IREAD)
    os.chmod(here + "\\locked", stat.S_IREAD)


def deep():
    path = here
    for _ in range(40):
        path += "\\" + "d" * 20
        os.mkdir(path)
    write(path + "\\bottom.txt")


def shut_out():
    write(here + "\\shut.txt")
    result = subprocess.run(["icacls", "shut.txt", "/inheritance:r", "/deny", "*S-1-1-0:F"], capture_output=True, text=True,
                            creationflags=0x08000000)
    if result.returncode:
        raise OSError(f"icacls exit {result.returncode}: {(result.stdout + result.stderr).strip()[:120]}")


attempt("read-only folder", read_only_folder)
attempt("reserved name", lambda: write(here + "\\con"))
attempt("trailing dot", lambda: write(here + "\\dotted."))
attempt("trailing space", lambda: write(here + "\\spaced "))
attempt("past 260 characters", deep)
attempt("marked system", lambda: (write(here + "\\marked.txt"),
                                  subprocess.run(["attrib", "+s", "+h", "+r", "marked.txt"], creationflags=0x08000000)))
attempt("shut the user out", shut_out)
print("worked: " + ", ".join(done), flush=True)
sys.exit(0)
