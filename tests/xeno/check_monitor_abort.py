"""The child must exit itself on a sampled display-VRAM breach."""

import os
import subprocess
import sys


def main():
    exe, model = sys.argv[1:3]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='1,0')
    result = subprocess.run([exe, model, '0', '7', '-2', '--runner-monitor-abort'],
                            capture_output=True, text=True, timeout=20, env=env)
    marker = 'secondary display VRAM reserve failed'
    if result.returncode != 3 or marker not in result.stderr:
        print(f'expected exit 3 and marker; got exit {result.returncode}')
        print(result.stdout[-1000:])
        print(result.stderr[-1000:])
        return 1
    print('secondary monitor: injected low reading ended its own process with exit 3')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
