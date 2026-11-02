"""Not a breach by itself: the listening side of the box-to-box test. It waits on a loopback port and writes the number to
port.txt. With `pair` it also starts a connecting child inside its own box, the control that shows loopback inside one box
works, so a refusal between two boxes is the wall and not a broken listener."""
import socket
import subprocess
import sys

server = socket.socket()
server.bind(("127.0.0.1", 0))
server.listen(1)
server.settimeout(float(sys.argv[1]))
port = server.getsockname()[1]
with open("port.tmp", "w", encoding="utf-8") as out:
    out.write(str(port))
__import__("os").replace("port.tmp", "port.txt")
if len(sys.argv) > 2:
    subprocess.Popen([sys.executable, "tcp_connect.py", "127.0.0.1", str(port)], creationflags=0x08000000)
try:
    server.accept()
    print("accepted")
except TimeoutError:
    print("nobody came")
