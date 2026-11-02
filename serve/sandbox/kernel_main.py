"""The real Python of the sandbox: one process per turn, run inside the isolation box, never in the server.

A line of JSON in, {"code": ...}; a line of JSON out, {"output": ..., "ok": ...}. Names stay between blocks of a turn, because
importing sympy again for each block takes half a second. Standard input holds only what the program gave `feed`.
This file imports nothing from the repository, so the box needs no access to it beyond this one file.
"""
import contextlib
import io
import json
import sys
import traceback

REAL_IN, REAL_OUT = sys.stdin, sys.stdout


def _approx(got, want):
    if isinstance(got, float) or isinstance(want, float):
        try:
            return abs(float(got) - float(want)) < 1e-6
        except (TypeError, ValueError):
            return False
    return False


def expect(got, want):
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


def main():
    feed = Feed()
    space = {'__name__': '__main__', 'expect': expect, 'feed': feed.add}
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
