"""The real-Python kernel inside a box, spoken to as the engine speaks to it. Windows or Linux with a backend; the code here is the test's own."""
import base64

import pytest

from serve.sandbox import exec_bridge
from serve.sandbox.exec_bridge import ExecBridge
from serve.sandbox.isolation import Isolation, Policy

POLICY = Policy(memory_bytes=1 << 30, seconds=60, processes=12, disk_bytes=64 << 20)


def block(bridge, code, fresh=False):
    answer = bridge.handle(f"SBX {int(fresh)} {base64.b64encode(code.encode('utf-8')).decode('ascii')}\n").split()
    return answer[1] == '1', base64.b64decode(answer[2]).decode('utf-8') if len(answer) > 2 else ''


@pytest.fixture
def bridge():
    made = ExecBridge(lambda: Isolation.start(POLICY))
    yield made
    made.close()


def test_names_stay_between_blocks(bridge):
    assert block(bridge, 'x = 40\n', fresh=True) == (True, '')
    assert block(bridge, 'print(x + 2)\n') == (True, '42\n')


def test_expect_and_feed_are_there(bridge):
    ok, text = block(bridge, 'feed("3\\n")\nexpect(int(input()) * 2, 6)\nexpect(1 == 2)\n', fresh=True)
    assert ok and text == 'match: 6\nMISMATCH: expected True, got False\n'


def test_the_libraries_are_there(bridge):
    ok, text = block(bridge, 'import sympy\nprint(sympy.factorint(360))\n', fresh=True)
    assert ok and text.strip() == '{2: 3, 3: 2, 5: 1}'


def test_an_error_is_the_block_s_result(bridge):
    ok, text = block(bridge, '1 / 0\n', fresh=True)
    assert not ok and 'ZeroDivisionError' in text
    assert block(bridge, 'print(5)\n') == (True, '5\n'), 'and the kernel goes on'


def test_the_real_project_stays_out(bridge):
    ok, text = block(bridge, f'print(open({exec_bridge.__file__!r}).read()[:9])\n', fresh=True)
    assert not ok and 'PermissionError' in text


def test_a_block_that_never_ends_is_stopped(bridge, monkeypatch):
    monkeypatch.setattr(exec_bridge, 'BLOCK_SECONDS', 3.0)
    block(bridge, 'x = 1\n', fresh=True)
    ok, text = block(bridge, 'while True:\n    pass\n')
    assert not ok and 'ended the sandbox' in text
    assert block(bridge, 'print(5)\n') == (True, '5\n'), 'the next block gets a new kernel, not the dead one\'s broken pipe'


def test_a_dead_kernel_says_so_and_never_raises(monkeypatch):
    """The bench's tool loop calls a kernel directly: lm1 and htl died of a BrokenPipeError on the call after a block ran past its time."""
    monkeypatch.setattr(exec_bridge, 'BLOCK_SECONDS', 3.0)
    kernel = exec_bridge.Kernel(Isolation.start(POLICY))
    try:
        assert not kernel.run('while True:\n    pass\n')[0]
        assert not kernel.alive()
        ok, text = kernel.run('print(1)\n')
        assert not ok and 'ended' in text
    finally:
        kernel.close()


def test_a_new_turn_gets_a_new_kernel_and_a_new_box(bridge):
    block(bridge, 'x = 1\n(__import__("pathlib").Path("left.txt")).write_text("x")\n', fresh=True)
    first = bridge.kernel.box
    ok, text = block(bridge, 'import os\nprint("x" in globals(), os.path.exists("left.txt"))\n', fresh=True)
    assert ok and text == 'False False\n'
    assert bridge.kernel.box is not first
