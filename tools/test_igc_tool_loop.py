"""The bench's reading of the model's own tool format: python tools/test_igc_tool_loop.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.igc.external_bench import TOOL_HINT, tool_call

CALL = '<tool_call>\n<function=run_python>\n<parameter=code>\nprint(1 + 1)\nprint(2)\n</parameter>\n</function>\n</tool_call>'
assert tool_call('thinking</think>\n\n' + CALL) == 'print(1 + 1)\nprint(2)'
assert tool_call(CALL) == 'print(1 + 1)\nprint(2)', 'no thinking at all'
assert tool_call('thinking</think>\n\nThe answer is 3.') is None
assert tool_call('a call in the thinking ' + CALL + '</think>\n\nThe answer is 3.') is None, 'only the reply calls'
assert tool_call('first</think>' + CALL + ' again</think>' + 'done') is None, 'the last reply decides'
assert tool_call('</think>\n\n<tool_call>\n<function=other>\n<parameter=code>\nx\n</parameter>\n</function>\n</tool_call>') is None, 'another function'
assert tool_call('</think>\n\n' + CALL.replace('print(1 + 1)\nprint(2)', 'a = "</tool_call>"\nprint(a)')) == 'a = "</tool_call>"\nprint(a)', 'a closing tag in the code'
assert '<tool_call>' in TOOL_HINT and '"run_python"' in TOOL_HINT
print('ok')
