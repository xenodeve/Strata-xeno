"""Start ab.py fully detached (survives the calling shell): python launch_ab.py <ab.py args...>"""
import os, subprocess, sys
T = os.environ["TEMP"]
name = sys.argv[1]
log = open(os.path.join(T, name + ".out"), "w")
p = subprocess.Popen([sys.executable, "ab.py"] + sys.argv[1:], cwd=os.path.dirname(os.path.abspath(__file__)),
                     stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                     creationflags=0x00000008 | 0x00000200 | 0x08000000)  # DETACHED | NEW_PROCESS_GROUP | NO_WINDOW
print(p.pid)
