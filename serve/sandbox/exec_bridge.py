"""What runs a block the interpreter refused: the engine prints `SBX <fresh> <base64>`, this answers `SBXR <ok> <base64>`.

The kernel lives in an isolation box. The only way out of that rule is `STRATA_SANDBOX_THROWAWAY=1` off Windows, for a VM that is
thrown away. Anything else with no box answers that the block cannot run, so nothing the model wrote runs bare on the user's PC.

The pipe to the kernel is also the program's only way to an MCP tool: while a block runs it asks, and this side answers (`mcp` in
kernel_main.py, serve/mip/mcp_adapter.py). The connections, the credentials and every check stay here.
"""
import base64
import json
import os
import subprocess
import sys
from pathlib import Path

KERNEL = Path(__file__).with_name('kernel_main.py')
BLOCK_SECONDS = 120.0


def throwaway() -> bool:
    return os.name != 'nt' and os.environ.get('STRATA_SANDBOX_THROWAWAY') == '1'


def kernel_env() -> dict:
    """What Python needs and nothing more: a key can sit in the server's environment, and the kernel runs the model's code."""
    return {k: os.environ[k] for k in ('PATH', 'HOME', 'LANG', 'SYSTEMROOT', 'TEMP', 'TMP', 'TMPDIR') if k in os.environ}


class Kernel:
    def __init__(self, box=None):
        self.box = box
        argv = [sys.executable, '-I', '-B', str(KERNEL)]
        if box is not None:
            self.proc = box.popen(argv)
        elif throwaway():
            self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding='utf-8', env=kernel_env())
        else:
            raise RuntimeError('no isolation box')
        ready = self.proc.stdout.readline()
        if not ready:
            raise RuntimeError('the kernel did not start')

    def send(self, message: dict):
        self.proc.stdin.write(json.dumps(message) + '\n')
        self.proc.stdin.flush()

    def run(self, code: str, tools=None) -> tuple[bool, str]:
        self.send({'code': code})
        while True:   # the program's tool calls, until the block's final reply
            line = self.proc.stdout.readline()
            if not line:
                return False, '[the block ended the sandbox]'
            reply = json.loads(line)
            if 'mcp' not in reply:
                return reply['ok'], reply['output']
            self.send({'mcp': answer(tools, reply['mcp'])})

    def close(self):
        self.proc.kill()
        self.proc.wait()


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
    may reach (serve/mip/mcp_adapter.py): a request with no MCP servers hands over nothing, and its blocks run as they always did."""

    def __init__(self, box_factory=None):
        self.box_factory = box_factory
        self.kernel = None
        self.tools = None
        self.turn = object()

    def begin(self, tools=None):
        self.tools, self.turn = tools, object()

    def stop(self):
        """The engine stops waiting for a block at STOP, and to it an answer that comes later is a bad command: the next request's error."""
        self.turn = None

    def handle(self, line: str) -> str:
        """The SBXR line for this block, or '' when nothing is waiting for one."""
        tools, turn = self.tools, self.turn   # a block that outlives its request keeps that request's reach, never the next one's
        if turn is None:
            return ''
        parts = line.split()
        fresh, code = parts[1] == '1', base64.b64decode(parts[2]).decode('utf-8', 'replace') if len(parts) > 2 else ''
        if fresh and self.kernel:
            self.kernel.close()
            self.kernel = None
        try:
            if self.kernel is None:
                self.kernel = Kernel(self.box_factory() if self.box_factory else None)
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

    def close(self):
        if self.kernel:
            self.kernel.close()
            self.kernel = None
