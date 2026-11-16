"""Strata Desktop's log, `desktop.log` in the data folder (#310 part 3): rotated at 1 MB, three old files kept.

Every token the app holds is registered with `add_secret` and blanked as `***` in every line written: the message,
its arguments, a traceback's text - for every logger, a library's (pywebview) included, because the scrubbing is done
on the formatted line by the file handler itself.  `install_hooks` sends an unhandled exception of the main thread or
any other thread here (the traceback's text only: no locals), `say` is what `serve.runner._say` is set to, and
`capture_stdio` sends what is printed (the windowed exe has no console; MCP servers print their lines) here too.
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import sys
import threading
from pathlib import Path

LOG_BYTES = 1 << 20
LOG_FILES = 3
FORMAT = "%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s"

_lock = threading.Lock()
_secrets: list[str] = []                 # every form of every token, longest first
_handler: logging.Handler | None = None


def add_secret(s: str) -> None:
    """Blank `s` from now on: as it is, and as JSON writes it once and twice (a token with `"` or `\\` inside a script
    or a JSON body), and as `repr` writes it.  An empty value (an unreadable token reads as "") is ignored."""
    if not isinstance(s, str) or not s:
        return
    forms = {s, json.dumps(s)[1:-1], json.dumps(json.dumps(s))[1:-1], repr(s)[1:-1]}
    with _lock:
        _secrets.extend(f for f in forms if f and f not in _secrets)
        _secrets.sort(key=len, reverse=True)


def forget_secrets() -> None:
    """For tests."""
    with _lock:
        _secrets.clear()


def scrub(text: str) -> str:
    with _lock:
        secrets = list(_secrets)
    for s in secrets:
        if s in text:
            text = text.replace(s, "***")
    return text


class ScrubFormatter(logging.Formatter):
    """A formatter whose whole output - message, arguments, traceback, stack - has every secret blanked."""

    def format(self, record: logging.LogRecord) -> str:
        return scrub(super().format(record))


class ScrubFilter(logging.Filter):
    """Blanks the secrets in a record's message and arguments before any formatter sees it (the formatter blanks the
    traceback's text as well)."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:                # a message whose arguments do not fit: keep its template only
            msg = str(record.msg)
        record.msg, record.args = scrub(msg), None
        return True


def setup(path: Path) -> logging.Logger:
    """The rotating file handler on the root logger (replacing one an earlier `setup` added), level INFO.  The file
    is opened at the first line (`delay`), so a second start that exits at once never holds it."""
    global _handler
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(path, maxBytes=LOG_BYTES, backupCount=LOG_FILES,
                                                   encoding="utf-8", delay=True)
    handler.setFormatter(ScrubFormatter(FORMAT))
    handler.addFilter(ScrubFilter())
    root = logging.getLogger()
    close()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    _handler = handler
    return logging.getLogger("strata.desktop")


def close() -> None:
    """Removes and closes the handler `setup` added (Quit, and tests)."""
    global _handler
    if _handler is not None:
        logging.getLogger().removeHandler(_handler)
        _handler.close()
        _handler = None


def say(text: str) -> None:
    """`serve.runner._say` in the app: the runner's lines go to the log."""
    logging.getLogger("strata.desktop.runner").info("%s", text)


def _excepthook(exc_type, exc, tb) -> None:
    logging.getLogger("strata.desktop").error("unhandled %s", exc_type.__name__, exc_info=(exc_type, exc, tb))


def _thread_excepthook(args) -> None:
    if args.exc_type is SystemExit:
        return
    name = args.thread.name if args.thread is not None else "?"
    logging.getLogger("strata.desktop").error("unhandled %s in thread %s", args.exc_type.__name__, name,
                                              exc_info=(args.exc_type, args.exc_value, args.exc_traceback))


def install_hooks() -> None:
    """Unhandled exceptions of the main thread and of every other thread go to the log, scrubbed."""
    sys.excepthook = _excepthook
    threading.excepthook = _thread_excepthook


class _LogWriter:
    """A text stream that logs each complete line (and what `flush` finds left)."""

    encoding = "utf-8"
    errors = "replace"

    def __init__(self, logger: logging.Logger, level: int):
        self._logger, self._level = logger, level
        self._buf = ""
        self._lock = threading.Lock()
        self._inside = threading.local()     # a failing handler writes to stderr: never loop back into the log

    def write(self, s: str) -> int:
        if not isinstance(s, str) or getattr(self._inside, "on", False):
            return len(s) if isinstance(s, str) else 0
        with self._lock:
            self._buf += s
            *lines, self._buf = self._buf.split("\n")
        self._emit(lines)
        return len(s)

    def flush(self) -> None:
        with self._lock:
            rest, self._buf = self._buf, ""
        self._emit([rest])

    def _emit(self, lines: list[str]) -> None:
        self._inside.on = True
        try:
            for line in lines:
                if line.strip():
                    self._logger.log(self._level, "%s", line.rstrip("\r"))
        finally:
            self._inside.on = False

    def isatty(self) -> bool:
        return False

    def writable(self) -> bool:
        return True


def capture_stdio() -> None:
    """`sys.stdout` and `sys.stderr` into the log (INFO and WARNING), line by line."""
    sys.stdout = _LogWriter(logging.getLogger("strata.desktop.stdout"), logging.INFO)
    sys.stderr = _LogWriter(logging.getLogger("strata.desktop.stderr"), logging.WARNING)
