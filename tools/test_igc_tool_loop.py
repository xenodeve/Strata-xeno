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

# --hint-after: the hint opens the call, the model writes the program and closes it; the bench reads the two together
from tools.igc.external_bench import HINT
WRITTEN = 'print(6 * 7)\n</parameter>\n</function>\n</tool_call>'
assert tool_call(HINT + WRITTEN) == 'print(6 * 7)', 'the hint closes the thinking: a reply call, answered in the tool turn'
assert called_in_thinking(HINT + WRITTEN) is None, 'never answered inside the thinking (4 of 22 rows of m5h2 ended the turn)'
assert tool_call(WRITTEN) is None, 'without the hint the open is missing'
assert HINT.count('<tool_call>') == 1 and HINT.index('</think>') < HINT.index('<tool_call>') and 'Python here is a good idea' in HINT

# --hint-at wait: only after the model's own Wait / Alternatively opening a sentence, and the hint goes on from its word
from tools.igc.external_bench import HINT_WAIT, hint_point
assert hint_point('wait', 'So the area is 1/4.\n\nWait') and hint_point('wait', 'gives 466 m. Alternatively')
assert not hint_point('wait', 'we wait') and not hint_point('wait', 'Waiting') and not hint_point('wait', 'so Wait')
assert not hint_point('wait', 'Wait, no.\n'), 'a sentence already past its Wait'
assert hint_point('line', 'a line\n') and not hint_point('line', 'a line')
assert tool_call('\n\nWait' + HINT_WAIT + WRITTEN) == 'print(6 * 7)' and HINT_WAIT.startswith(', I can use Python')

# One row's engine lines: m5t's loop rule closed the thinking on 2 rows, and nothing in the records said which (#288)
import tempfile
from tools.igc.external_bench import read_events
with tempfile.TemporaryDirectory() as d:
    log = Path(d) / 'engine.log'
    log.write_text('strata serve: request metrics: row 1\n', encoding='utf-8')
    mark = log.stat().st_size
    with open(log, 'a', encoding='utf-8') as f:
        f.write('strata IGC learned audit: head_status=4294967295 record=00\nstrata igc loop: counting, closed the thinking\n'
                'strata serve: request metrics: a\nstrata igc sandbox: ran a block\nstrata IGC learned audit: head_status=3 record=00\n'
                'strata serve: request metrics: b\n')
    got = read_events(log, mark)
    assert got == dict(requests=2, rules=['loop: counting, closed the thinking', 'sandbox: ran a block'], heads=['3']), got
    assert read_events(Path(d) / 'none.log', 0) is None
    assert read_events(log, mark, mark + 60)['rules'] == [], "a row ends at the next row's offset"

    # the report says what ran: the command, the code, the bundle and the tokenizer bytes
    from tools.igc.external_bench import provenance
    (Path(d) / 'tokenizer').mkdir()
    (Path(d) / 'tokenizer' / 'vocab.json').write_bytes(b'{}')
    p = provenance(None, d)
    assert set(p) == {'argv', 'git', 'bundle_sha256', 'tokenizer_sha256'} and p['bundle_sha256'] is None, p
    assert p['tokenizer_sha256']['vocab.json'] == '44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a', p
print('ok')
