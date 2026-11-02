"""The real Python of the sandbox: one process per turn, run inside the isolation box, never in the server.

A line of JSON in, {"code": ...}; a line of JSON out, {"output": ..., "ok": ...}. Names stay between blocks of a turn, because
importing sympy again for each block takes half a second. Standard input holds only what the program gave `feed`.
This file imports nothing from the repository, so the box needs no access to it beyond this one file.

`mcp` is the program's way to MCP tools, and the pipe is its only way out of the box: while a block runs, a line {"mcp": ...} goes
up and the answer comes down, {"mcp": ...} too, before the block's final reply. The server holds the connections, the credentials
and the rules; nothing here is trusted, and a program can write anything down this pipe.
"""
import contextlib
import io
import json
import sys
import threading
import traceback

REAL_IN, REAL_OUT = sys.stdin, sys.stdout
PIPE = threading.Lock()


def _approx(got, want):
    if isinstance(got, float) or isinstance(want, float):
        try:
            return abs(float(got) - float(want)) < 1e-6
        except (TypeError, ValueError):
            return False
    return False


def expect(got, want=True):
    """A result set against what the question says it should be."""
    if got == want or _approx(got, want):
        print(f'match: {got!r}')
    else:
        print(f'MISMATCH: expected {want!r}, got {got!r}')


class Feed(io.StringIO):
    """stdin: what `feed` was given, read from where the program left off."""

    def add(self, text):
        at = self.tell()
        self.seek(0, io.SEEK_END)
        self.write(text)
        self.seek(at)


class ToolError(Exception):
    """A tool that did not run, or ran and failed. Its message is the reason: the model reads it as the block's error."""


def ask(request):
    with PIPE:   # the answers carry no id, so one question at a time
        REAL_OUT.write(json.dumps({'mcp': request}) + '\n')
        REAL_OUT.flush()
        line = REAL_IN.readline()
    if not line:
        raise ToolError('fault: the server closed the pipe')
    result = json.loads(line)['mcp']
    if result['status'] != 'success':
        raise ToolError(f"{result['status']}: {result['typed_content']}")
    return result['typed_content']


class Tool:
    def __init__(self, server, name):
        self.server, self.name = server, name

    def __call__(self, /, *args, **arguments):
        if args:
            raise TypeError(f'name the arguments: mcp.{self.server}.{self.name}(arg=value)')
        return ask({'op': 'call', 'server': self.server, 'tool': self.name, 'arguments': arguments})


class Server:
    def __init__(self, name):
        self._name = name

    def __getattr__(self, tool):
        if tool.startswith('__'):
            raise AttributeError(tool)
        return Tool(self._name, tool)


class Mcp:
    """`mcp.tools("words")` lists the tools that match, with their signatures. `mcp.<server>.<tool>(arg=value)` calls one and gives a
    dict or a list for a structured result, else a str: only what the program prints reaches the reasoning."""

    def tools(self, query=''):
        return ask({'op': 'tools', 'query': str(query)})

    def __getattr__(self, server):
        if server.startswith('__'):
            raise AttributeError(server)
        return Server(server)


def main():
    feed = Feed()
    space = {'__name__': '__main__', 'expect': expect, 'feed': feed.add, 'mcp': Mcp()}
    sys.stdin = feed
    for module in ('sympy', 'numpy', 'mpmath', 'networkx'):
        try:
            __import__(module)
        except ImportError:
            pass
    print(json.dumps({'ready': True}), file=REAL_OUT, flush=True)
    for line in REAL_IN:
        code = json.loads(line)['code']
        out = io.StringIO()
        ok = True
        with contextlib.redirect_stdout(out):
            try:
                exec(compile(code, '<block>', 'exec'), space)
            except SystemExit:
                pass
            except BaseException:
                ok = False
                last = traceback.format_exc().strip().split('\n')
                where = [x.strip() for x in last if '<block>' in x][-1:]
                print((where[0] + '\n' if where else '') + last[-1])
        print(json.dumps({'output': out.getvalue(), 'ok': ok}), file=REAL_OUT, flush=True)


if __name__ == '__main__':
    main()
