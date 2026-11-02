"""What runs a block the interpreter refused: the engine prints `SBX <fresh> <base64>`, this answers `SBXR <ok> <base64>`.

The kernel lives in an isolation box. The only way out of that rule is `STRATA_SANDBOX_THROWAWAY=1` off Windows, for a VM that is
thrown away. Anything else with no box answers that the block cannot run, so nothing the model wrote runs bare on the user's PC.

The pipe to the kernel is also the program's only way to an MCP tool: while a block runs it asks, and this side answers (`mcp` in
kernel_main.py, serve/mip/mcp_adapter.py). The connections, the credentials and every check stay here.
"""
import base64
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

KERNEL = Path(__file__).with_name('kernel_main.py')
BLOCK_SECONDS = 120.0


def throwaway() -> bool:
    return os.name != 'nt' and os.environ.get('STRATA_SANDBOX_THROWAWAY') == '1'


def kernel_env() -> dict:
    """What Python needs and nothing more: a key can sit in the server's environment, and the kernel runs the model's code."""
    return {k: os.environ[k] for k in ('PATH', 'HOME', 'LANG', 'LC_ALL', 'LC_CTYPE', 'LD_LIBRARY_PATH', 'SYSTEMROOT', 'TEMP', 'TMP', 'TMPDIR')
            if k in os.environ}


class Kernel:
    def __init__(self, box=None):
        self.box = box
        self.ended = False
        if box is not None:   # the box reads no repository: the kernel goes in as a copy of its one file
            shutil.copyfile(KERNEL, Path(box.workdir) / KERNEL.name)
            self.proc = box.popen(['python', '-I', '-B', KERNEL.name])
        elif throwaway():
            argv = [sys.executable, '-I', '-B', str(KERNEL)]
            self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding='utf-8', env=kernel_env())
        else:
            raise RuntimeError('no isolation box')
        ready = self.proc.stdout.readline()
        if not ready:
            raise RuntimeError('the kernel did not start')

    def alive(self) -> bool:
        """False once a run ended it (a stop, its time, its own exit): the next call starts another, as a fresh block does."""
        poll = getattr(self.proc, 'poll', None)
        return not self.ended and (poll is None or poll() is None)

    def send(self, message: dict):
        self.proc.stdin.write(json.dumps(message) + '\n')
        self.proc.stdin.flush()

    def run(self, code: str, tools=None, cancel=None) -> tuple[bool, str]:
        timer = threading.Timer(BLOCK_SECONDS, self.proc.kill)   # tool calls count: the decoding waits on all of it
        timer.start()
        ended = threading.Event()
        if cancel is not None:   # a call in the reply: the user's Stop ends it, as it ends any tool (the engine's blocks have the engine's STOP)
            def watch():
                while not ended.wait(0.1):
                    if cancel.is_set():
                        self.proc.kill()
                        return
            threading.Thread(target=watch, daemon=True, name="kernel-stop").start()
        try:
            self.send({'code': code})
            while True:   # the program's tool calls, until the block's final reply
                line = self.proc.stdout.readline()
                if not line:
                    self.ended = True
                    return False, f'[the block ended the sandbox, or ran past {BLOCK_SECONDS:.0f} s]'
                reply = json.loads(line)
                if 'mcp' not in reply:
                    return reply['ok'], reply['output']
                self.send({'mcp': answer(tools, reply['mcp'])})
        finally:
            timer.cancel()
            ended.set()

    def close(self):
        self.proc.kill()
        self.proc.wait()
        for pipe in (getattr(self.proc, 'stdin', None), getattr(self.proc, 'stdout', None)):
            try:
                if pipe is not None:
                    pipe.close()
            except OSError:
                pass
        if self.box is not None:
            self.box.kill()


def answer(tools, request) -> dict:
    """Always an answer: the kernel waits for one, and decoding waits for the kernel."""
    if tools is None:
        return {'status': 'unsupported', 'typed_content': 'this request has no MCP servers'}
    try:
        return tools.answer(request)
    except Exception as error:  # noqa: BLE001 - only its name: an exception can quote what must stay on this side of the box
        return {'status': 'fault', 'typed_content': f'the call failed ({type(error).__name__})'}


class ExecBridge:
    """One kernel per turn. `box_factory` gives a new isolation box, or None where there is none. `begin` says what a request's blocks
    may reach (serve/mip/mcp_adapter.py): a request with no MCP servers hands over nothing, and its blocks run as they always did.

    The reasoning tool (serve/reasoning_tool.py) also runs a request's `run_python` calls in its reply here (`run_reply`), in the kernel its
    thinking used, so that the names made there are still there. `owner` is that request's mark, None for every other request."""

    def __init__(self, box_factory=None):
        self.box_factory = box_factory
        self.kernel = None
        self.tools = None
        self.turn = object()
        self.owner = None              # the request on the reasoning tool's route whose engine turn this is
        self.kernel_owner = None       # whose names the kernel holds
        self.side = {}                 # a request's own kernel for its calls in the reply, made while another request held the engine
        self.lock = threading.Lock()   # one program at a time in a kernel: a block comes on the pump's thread, a call in the reply on the request's

    def begin(self, tools=None, owner=None):
        self.tools, self.owner, self.turn = tools, owner, object()

    def stop(self):
        """The engine stops waiting for a block at STOP, and to it an answer that comes later is a bad command: the next request's error."""
        self.turn = None

    def _box(self):
        return self.box_factory() if self.box_factory else None

    def handle(self, line: str) -> str:
        """The SBXR line for this block, or '' when nothing is waiting for one."""
        tools, turn, owner = self.tools, self.turn, self.owner   # a block that outlives its request keeps that request's reach, never the next one's
        if turn is None:
            return ''
        parts = line.split()
        fresh, code = parts[1] == '1', base64.b64decode(parts[2]).decode('utf-8', 'replace') if len(parts) > 2 else ''
        with self.lock:
            if fresh and self.kernel:
                self.kernel.close()
                self.kernel = None
            try:
                if self.kernel is None:
                    self.kernel = Kernel(self._box())
                self.kernel_owner = owner
                if tools is not None:
                    if fresh:
                        tools.begin_turn()
                    tools.begin_block(BLOCK_SECONDS)
                ok, text = self.kernel.run(code, tools)
            except Exception as error:  # noqa: BLE001 - the model reads this as the block's result
                ok, text = False, f'[the sandbox could not run this: {error}]'
        if turn is not self.turn:
            return ''
        return f"SBXR {int(ok)} {base64.b64encode(text.encode('utf-8')).decode('ascii')}\n"

    def run_reply(self, owner, code: str, tools=None, cancel=None) -> tuple[bool, str]:
        """A `run_python` call in `owner`'s reply -> (ok, output). It runs where this request's names are: the kernel its thinking moved to,
        while that is still its own. With the engine free, a request whose thinking stayed in the engine's interpreter takes the kernel
        fresh; the engine's next block starts its own (`fresh`), so no request reads another's names. While another request holds the
        engine, the call gets a kernel of this request's own, kept for its later calls until `end`."""
        side = self.side.get(owner)
        if side is not None and not side.alive():   # a run that was stopped or ran past its time took the kernel with it
            side.close()
            del self.side[owner]
            side = None
        if side is None:
            with self.lock:
                mine = self.kernel is not None and self.kernel_owner is owner and self.kernel.alive()
                if mine or self.turn is None:
                    try:
                        if not mine:
                            if self.kernel:
                                self.kernel.close()
                            self.kernel = None
                            self.kernel = Kernel(self._box())
                            self.kernel_owner = owner
                            if tools is not None:
                                tools.begin_turn()
                        if tools is not None:
                            tools.begin_block(BLOCK_SECONDS)
                        return self.kernel.run(code, tools, cancel)
                    except Exception as error:  # noqa: BLE001 - the model reads this as the call's result
                        return False, f'[the sandbox could not run this: {error}]'
        try:
            if side is None:
                side = self.side[owner] = Kernel(self._box())
                if tools is not None:
                    tools.begin_turn()
            if tools is not None:
                tools.begin_block(BLOCK_SECONDS)
            return side.run(code, tools, cancel)
        except Exception as error:  # noqa: BLE001
            return False, f'[the sandbox could not run this: {error}]'

    def end(self, owner):
        """`owner`'s request is over: its kernels end with it, and no later request starts with its names."""
        side = self.side.pop(owner, None)
        if side is not None:
            side.close()
        with self.lock:
            if self.kernel is not None and self.kernel_owner is owner:
                self.kernel.close()
                self.kernel = None
                self.kernel_owner = None

    def close(self):
        if self.kernel:
            self.kernel.close()
            self.kernel = None
        for side in list(self.side.values()):
            side.close()
        self.side.clear()
