"""The bench's reading of the model's own tool format: python tools/test_igc_tool_loop.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.igc.external_bench import TOOL_HINT, called_in_thinking, tool_call

CALL = '<tool_call>\n<function=run_python>\n<parameter=code>\nprint(1 + 1)\nprint(2)\n</parameter>\n</function>\n</tool_call>'
assert tool_call('thinking</think>\n\n' + CALL) == 'print(1 + 1)\nprint(2)'
assert tool_call(CALL) == 'print(1 + 1)\nprint(2)', 'no thinking at all'
assert tool_call('thinking</think>\n\nThe answer is 3.') is None
assert tool_call('a call in the thinking ' + CALL + '</think>\n\nThe answer is 3.') is None, 'only the reply calls'
assert tool_call('first</think>' + CALL + ' again</think>' + 'done') is None, 'the last reply decides'
assert tool_call('</think>\n\n<tool_call>\n<function=other>\n<parameter=code>\nx\n</parameter>\n</function>\n</tool_call>') is None, 'another function'
assert tool_call('</think>\n\n' + CALL.replace('print(1 + 1)\nprint(2)', 'a = "</tool_call>"\nprint(a)')) == 'a = "</tool_call>"\nprint(a)', 'a closing tag in the code'
assert '<tool_call>' in TOOL_HINT and '"run_python"' in TOOL_HINT
JSON = '<tool_call>\n{"name":"run_python","arguments":{"code":"print(3)\\nprint(4)"}}\n</tool_call>'
assert tool_call('x</think>\n' + JSON) == 'print(3)\nprint(4)', 'the JSON form the model also writes'
assert tool_call('x</think>\n<tool_call>\n{"name":"other","arguments":{"code":"x"}}\n</tool_call>') is None
assert tool_call('x</think>\n<tool_call>\n{not json}\n</tool_call>') is None
assert called_in_thinking("Let's call.\n\n" + CALL) == 'print(1 + 1)\nprint(2)', 'a call closed while thinking'
assert called_in_thinking("Let's call.\n\n" + JSON) == 'print(3)\nprint(4)'
assert called_in_thinking('done</think>\n\n' + CALL) is None, 'after the thinking it is a reply'
assert called_in_thinking('a call ' + CALL + ' and more thinking') is None, 'only one that just closed'
print('ok')
