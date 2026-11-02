"""What runs a block the interpreter refused: the engine prints `SBX <fresh> <base64>`, this answers `SBXR <ok> <base64>`.

The kernel lives in an isolation box. The only way out of that rule is `STRATA_SANDBOX_THROWAWAY=1` off Windows, for a VM that is
thrown away. Anything else with no box answers that the block cannot run, so nothing the model wrote runs bare on the user's PC.
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


class Kernel:
    def __init__(self, box=None):
        self.box = box
        argv = [sys.executable, '-I', '-B', str(KERNEL)]
        if box is not None:
            self.proc = box.popen(argv)
        elif throwaway():
            self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding='utf-8')
        else:
            raise RuntimeError('no isolation box')
        ready = self.proc.stdout.readline()
        if not ready:
            raise RuntimeError('the kernel did not start')

    def run(self, code: str) -> tuple[bool, str]:
        self.proc.stdin.write(json.dumps({'code': code}) + '\n')
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            return False, '[the block ended the sandbox]'
        reply = json.loads(line)
        return reply['ok'], reply['output']

    def close(self):
        self.proc.kill()
        self.proc.wait()


class ExecBridge:
    """One kernel per turn. `box_factory` gives a new isolation box, or None where there is none."""

    def __init__(self, box_factory=None):
        self.box_factory = box_factory
        self.kernel = None

    def handle(self, line: str) -> str:
        parts = line.split()
        fresh, code = parts[1] == '1', base64.b64decode(parts[2]).decode('utf-8', 'replace') if len(parts) > 2 else ''
        if fresh and self.kernel:
            self.kernel.close()
            self.kernel = None
        try:
            if self.kernel is None:
                self.kernel = Kernel(self.box_factory() if self.box_factory else None)
            ok, text = self.kernel.run(code)
        except Exception as error:  # noqa: BLE001 - the model reads this as the block's result
            ok, text = False, f'[the sandbox could not run this: {error}]'
        return f"SBXR {int(ok)} {base64.b64encode(text.encode('utf-8')).decode('ascii')}\n"

    def close(self):
        if self.kernel:
            self.kernel.close()
            self.kernel = None
