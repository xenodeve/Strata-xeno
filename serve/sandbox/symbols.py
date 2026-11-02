"""serve/sandbox/symbols.py - reading and changing the copy by symbol instead of by file (the design's L2, 4.9), with the names Serena uses.

Python goes through the `ast` module. Other languages go through the `tree_sitter` package when both it and that language's grammar can be imported; when not, the answer says that only text search is available,
because a symbol read off text with a pattern would be a guess, and an edit made on a guess lands in the wrong place. A file that does not parse is said so of, and search still finds what is in it.

A name that fits more than one symbol answers with all of them and changes nothing. Each answer carries a `ref` (file:Class/name@line) that names exactly one symbol. Edits go through the bytes of the file, so that
what is not the symbol stays as it was: line endings, a byte order mark, an encoding other than UTF-8."""
from __future__ import annotations

import ast
import codecs
import functools
import importlib
import io
import os
import re
import time
import tokenize
from dataclasses import dataclass

from serve.sandbox.search import GENERATED, Search, clean_path
from serve.sandbox.workspace import MTIME_TICK_NS, WorkspaceError

MAX_TEXT_HITS = 20
_LINE_END = re.compile(rb"\r\n|\r|\n")                                        # the ends Python's own reader counts; bytes.splitlines also breaks on form feed
_REF = re.compile(r"(?:(?P<file>[^:]+):(?!:))?(?P<name>.+?)(?:@(?P<line>\d+))?\Z", re.S)
_PARSE_ERRORS = (SyntaxError, ValueError, RecursionError, MemoryError, UnicodeError, LookupError)


@dataclass(frozen=True)
class Grammar:
    module: str
    call: str
    kinds: tuple[tuple[str, str], ...]                                        # node type -> what to call it


_C = (("function_definition", "function"), ("struct_specifier", "struct"), ("enum_specifier", "enum"), ("union_specifier", "union"))
_CPP = _C + (("class_specifier", "class"), ("namespace_definition", "namespace"))
_JS = (("function_declaration", "function"), ("generator_function_declaration", "function"), ("class_declaration", "class"), ("method_definition", "method"))
_TS = _JS + (("abstract_class_declaration", "class"), ("interface_declaration", "interface"), ("enum_declaration", "enum"), ("type_alias_declaration", "type"))
_GO = (("function_declaration", "function"), ("method_declaration", "method"), ("type_spec", "type"))
_RUST = (("function_item", "function"), ("struct_item", "struct"), ("enum_item", "enum"), ("trait_item", "trait"), ("impl_item", "impl"), ("mod_item", "module"))
_JAVA = (("class_declaration", "class"), ("interface_declaration", "interface"), ("enum_declaration", "enum"), ("method_declaration", "method"), ("constructor_declaration", "constructor"))
GRAMMARS = {".c": Grammar("tree_sitter_c", "language", _C),
            **dict.fromkeys((".h", ".hh", ".hpp", ".cc", ".cpp", ".cxx"), Grammar("tree_sitter_cpp", "language", _CPP)),
            **dict.fromkeys((".js", ".jsx", ".mjs"), Grammar("tree_sitter_javascript", "language", _JS)),
            ".ts": Grammar("tree_sitter_typescript", "language_typescript", _TS), ".tsx": Grammar("tree_sitter_typescript", "language_tsx", _TS),
            ".go": Grammar("tree_sitter_go", "language", _GO), ".rs": Grammar("tree_sitter_rust", "language", _RUST), ".java": Grammar("tree_sitter_java", "language", _JAVA)}
_NEEDS_BODY = {"struct", "enum", "union", "class"}                            # without a body these are a use of the type or a forward declaration


@functools.lru_cache(maxsize=None)
def _parser(grammar: Grammar):
    """A parser for this grammar, or None when the package or the grammar cannot be imported or do not fit each other."""
    try:
        ts = importlib.import_module("tree_sitter")
        return ts.Parser(ts.Language(getattr(importlib.import_module(grammar.module), grammar.call)()))
    except Exception:  # noqa: BLE001 - a wheel built for another version fails in ways of its own; each one means the same here
        return None


@dataclass(frozen=True)
class Symbol:
    path: tuple[str, ...]                  # the enclosing names and its own
    kind: str
    line: int                              # 1-based, where the symbol starts (a decorator, if it has one)
    end: int
    start: int                             # offsets in the document's buffer. `start` is the start of the line, so the text carries its indentation
    stop: int
    alone: bool                            # nothing else shares its lines, so that it can be changed as a whole
    sure: bool = True                      # the parser met no error inside it


@dataclass
class Doc:
    rel: str
    buf: bytes                             # UTF-8 for Python, which is what `ast` counts in; the file's own bytes for tree-sitter
    prefix: bytes                          # a byte order mark, kept out of `buf`
    encoding: str | None                   # for a Python file not in UTF-8: the bytes go back in this encoding
    symbols: list[Symbol] | None           # None: there is no structure, and `problem` says why
    problem: str | None
    broken: bool = False                   # tree-sitter reads past errors: this says it met one somewhere in the file

    @property
    def eol(self) -> bytes:
        crlf = self.buf.count(b"\r\n")
        return b"\r\n" if crlf > self.buf.count(b"\n") - crlf else b"\n"

    def fit(self, text: str) -> bytes:
        """New text in the file's own line ending, without the trailing newline: the caller decides what goes around it."""
        eol = self.eol
        return re.sub(rb"\r\n|\r|\n", lambda _m: eol, text.rstrip("\r\n").encode("utf-8"))

    def raw(self, buf: bytes) -> bytes:
        return self.prefix + (buf.decode("utf-8").encode(self.encoding) if self.encoding else buf)


def _python_doc(rel: str, raw: bytes) -> Doc:
    prefix = codecs.BOM_UTF8 if raw.startswith(codecs.BOM_UTF8) else b""
    try:
        encoding = tokenize.detect_encoding(io.BytesIO(raw).readline)[0]
        foreign = None if encoding in ("utf-8", "utf-8-sig") else encoding
        text = raw[len(prefix):].decode(foreign or "utf-8")
        tree = ast.parse(text)
    except _PARSE_ERRORS as e:
        where = f" at line {e.lineno}" if isinstance(e, SyntaxError) and e.lineno else ""
        reason = e.msg if isinstance(e, SyntaxError) else type(e).__name__
        return Doc(rel, raw, b"", None, None, f"{rel} does not parse{where}: {reason}. Only text search is available for it.")
    buf = text.encode("utf-8")
    if foreign and text.encode(foreign) != raw:
        return Doc(rel, raw, b"", None, None, f"{rel}: its encoding {foreign} does not round-trip, so it cannot be edited by symbol. Only text search is available for it.")
    starts = [0] + [m.end() for m in _LINE_END.finditer(buf)]

    def symbol(node, path: tuple[str, ...], kind: str, first: int) -> Symbol:
        stop = starts[node.end_lineno - 1] + node.end_col_offset
        after = buf[stop:starts[node.end_lineno] if node.end_lineno < len(starts) else len(buf)].strip()
        before = buf[starts[node.lineno - 1]:starts[node.lineno - 1] + node.col_offset]
        alone = not before.strip(b" \t\f") and (not after or after.startswith(b"#"))
        return Symbol(path, kind, first, node.end_lineno, starts[first - 1], stop, alone)

    out: list[Symbol] = []

    def walk(node, path: tuple[str, ...], in_function: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                is_class = isinstance(child, ast.ClassDef)
                kind = "class" if is_class else ("method" if path and not in_function else "function")
                first = child.decorator_list[0].lineno if child.decorator_list else child.lineno
                out.append(symbol(child, path + (child.name,), kind, first))
                walk(child, path + (child.name,), not is_class)
            elif isinstance(child, (ast.Assign, ast.AnnAssign)):
                if not in_function:
                    for target in (child.targets if isinstance(child, ast.Assign) else [child.target]):
                        if isinstance(target, ast.Name):
                            out.append(symbol(child, path + (target.id,), "variable", child.lineno))
            elif isinstance(child, (ast.stmt, ast.excepthandler, ast.match_case)):
                walk(child, path, in_function)

    try:
        walk(tree, (), False)
    except RecursionError:
        return Doc(rel, raw, b"", None, None, f"{rel} is nested too deep to read. Only text search is available for it.")
    return Doc(rel, buf, prefix, foreign, sorted(out, key=lambda s: (s.line, s.path)), None)


def _node_name(node, buf: bytes) -> str | None:
    target = node.child_by_field_name("name")
    if target is None:                                                        # a Rust impl block is named by its type
        target = node.child_by_field_name("type")
    if target is None:                                                        # C and C++ keep a function's name at the end of a chain of declarators
        target = node.child_by_field_name("declarator")
        while target is not None and target.child_by_field_name("declarator") is not None:
            target = target.child_by_field_name("declarator")
    return buf[target.start_byte:target.end_byte].decode("utf-8", "replace") if target is not None else None


def _tree_doc(rel: str, raw: bytes, grammar: Grammar, parser) -> Doc:
    kinds = dict(grammar.kinds)
    tree = parser.parse(raw)
    out: list[Symbol] = []
    stack = [(tree.root_node, ())]
    while stack:
        node, path = stack.pop()
        kind = kinds.get(node.type)
        if kind and (kind not in _NEEDS_BODY or node.child_by_field_name("body") is not None):
            name = _node_name(node, raw)
            if name:
                path = path + tuple(name.split("::"))
                outer = node.parent if node.parent is not None and node.parent.type == "template_declaration" else node
                start = raw.rfind(b"\n", 0, outer.start_byte) + 1
                line_end = raw.find(b"\n", node.end_byte)
                after = raw[node.end_byte:line_end if line_end >= 0 else len(raw)].lstrip(b"; \t").strip()          # the ; of `class A {};` is not another statement
                alone = not raw[start:outer.start_byte].strip() and (not after or after.startswith((b"//", b"/*", b"#")))
                out.append(Symbol(path, kind, outer.start_point[0] + 1, node.end_point[0] + 1, start, node.end_byte, alone, not node.has_error))
        stack.extend((child, path) for child in reversed(node.named_children))
    return Doc(rel, raw, b"", None, sorted(out, key=lambda s: (s.line, s.path)), None, tree.root_node.has_error)


def parse(rel: str, raw: bytes) -> Doc:
    ext = os.path.splitext(rel)[1].lower()
    if ext in (".py", ".pyi"):
        return _python_doc(rel, raw)
    grammar = GRAMMARS.get(ext)
    parser = _parser(grammar) if grammar else None
    if parser is None:
        why = f"{grammar.module} or tree_sitter cannot be imported here" if grammar else f"no grammar is known for {ext or 'files with no extension'}"
        return Doc(rel, raw, b"", None, None, f"{rel}: {why}, so it has no outline. Only text search is available for it.")
    return _tree_doc(rel, raw, grammar, parser)


@dataclass(frozen=True)
class Query:
    file: str | None
    parts: tuple[str, ...]
    line: int | None


def _query(symbol: str) -> Query | None:
    m = _REF.fullmatch(symbol.strip()) if isinstance(symbol, str) else None
    if not m:
        return None
    parts = tuple(x for x in m["name"].replace("::", "/").replace(".", "/").split("/") if x)
    return Query(clean_path(m["file"]) if m["file"] else None, parts, int(m["line"]) if m["line"] else None) if parts else None


class Symbols:
    def __init__(self, workspace, search: Search | None = None):
        self.ws = workspace
        self.search = search or Search(workspace)
        self._docs: dict[str, tuple[tuple[int, int], Doc]] = {}

    def _doc(self, rel: str, stamp: tuple[int, int]) -> Doc | None:
        hit = self._docs.get(rel)
        if hit and hit[0] == stamp:
            return hit[1]
        try:
            raw = self.ws.read_stamped(rel, stamp)
        except (OSError, WorkspaceError):
            return None
        doc = parse(rel, raw)
        if stamp[1] < time.time_ns() - MTIME_TICK_NS:                         # the rule of Workspace._hash_copy; here a stale answer would be offsets into another text
            self._docs[rel] = (stamp, doc)
        return doc

    def _load(self, path: str) -> Doc | dict:
        rel = clean_path(path)
        stamp = self.search.files().get(rel)
        if stamp is None:
            return {"ok": False, "error": f"no such file in the copy: {path}", "near": self.search.near(rel)}
        if stamp[0] > self.ws.max_file or self.search.text(rel, stamp) is None:
            return {"ok": False, "error": f"{rel} is binary or too big to read", "text_only": True}
        return self._doc(rel, stamp) or {"ok": False, "error": f"{rel} could not be read"}

    @staticmethod
    def _row(doc: Doc, sym: Symbol, body: bool) -> dict:
        name = "/".join(sym.path)
        row = {"ref": f"{doc.rel}:{name}@{sym.line}", "file": doc.rel, "name": name, "kind": sym.kind, "line": sym.line, "end": sym.end}
        if body:
            row["body"] = doc.buf[sym.start:sym.stop].decode("utf-8", "replace")
        return row

    def outline(self, path: str, depth: int | None = None) -> dict:
        """The symbols of one file. `depth`: how many levels of nesting, for a very large file (design case 60)."""
        doc = self._load(path)
        if isinstance(doc, dict):
            return doc
        if doc.symbols is None:
            return {"ok": False, "error": doc.problem, "text_only": True}
        return {"ok": True, "path": doc.rel, "symbols": [self._row(doc, s, False) for s in doc.symbols if depth is None or len(s.path) <= depth]}

    def _find(self, q: Query) -> tuple[list[tuple[Doc, Symbol]], list[dict], list[str]]:
        """(the symbols that fit, the lines of text that mention the name in files with no structure, why those have none)."""
        files = self.search.files()
        leaf = q.parts[-1]
        if q.file:
            files = {q.file: files[q.file]} if q.file in files else {}
        word = re.compile(rf"(?<!\w){re.escape(leaf)}(?!\w)")
        found: list[tuple[Doc, Symbol]] = []
        text_hits: list[dict] = []
        problems: list[str] = []
        for rel, stamp in sorted(files.items()):
            if GENERATED.search(rel) and not q.file:
                continue
            text = self.search.text(rel, stamp)
            if text is None or leaf not in text:
                continue
            doc = self._doc(rel, stamp)
            if doc is None:
                continue
            if doc.symbols is None:
                hits = self.search.hits_in(rel, text, word, MAX_TEXT_HITS - len(text_hits)) if len(text_hits) < MAX_TEXT_HITS else []
                if hits:
                    text_hits += hits
                    problems.append(doc.problem)
                continue
            found += [(doc, s) for s in doc.symbols if s.path[-len(q.parts):] == q.parts and (q.line is None or s.line == q.line)]
        return found, text_hits, problems

    def find_symbol(self, name: str, body: bool = False) -> dict:
        """Every symbol of that name, as `Class/name`, `Class.name` or a `ref`. `text_hits`: mentions in files with no structure here; they are text, not confirmed symbols."""
        q = _query(name)
        if q is None:
            return {"ok": False, "error": f"that is not a symbol name: {name!r}"}
        found, text_hits, problems = self._find(q)
        out: dict = {"ok": bool(found or text_hits), "matches": [self._row(doc, s, body) for doc, s in found], "text_hits": text_hits}
        if problems:
            out["note"] = " ".join(problems)
        if not out["ok"]:
            out["error"] = f"no symbol named {name} in the copy"
        return out

    def _edit(self, symbol: str, change) -> dict:
        q = _query(symbol)
        if q is None:
            return {"ok": False, "error": f"that is not a symbol name: {symbol!r}"}
        found, text_hits, problems = self._find(q)
        if not found:
            if text_hits:
                return {"ok": False, "error": "it was found only as text, so nothing was changed. " + " ".join(problems), "text_hits": text_hits, "text_only": True}
            return {"ok": False, "error": f"no symbol named {symbol} in the copy, so nothing was changed"}
        if len(found) > 1:
            return {"ok": False, "error": f"{len(found)} symbols fit {symbol}, so nothing was changed. Choose one by its ref.", "matches": [self._row(doc, s, False) for doc, s in found]}
        doc, sym = found[0]
        if not sym.alone:
            return {"ok": False, "error": f"{self._row(doc, sym, False)['ref']} shares a line with other code, so nothing was changed. Change it as text."}
        if not sym.sure:
            return {"ok": False, "error": f"the parser met an error inside {self._row(doc, sym, False)['ref']}, so nothing was changed. Change it as text."}
        if q.parts[-1].encode("utf-8") not in doc.buf[sym.start:sym.stop]:
            return {"ok": False, "error": f"the text of {self._row(doc, sym, False)['ref']} does not hold its name, so nothing was changed. Change it as text."}      # a macro can make a parser see a name that is not where it says
        try:
            raw = doc.raw(change(doc, sym))
        except UnicodeError as e:
            return {"ok": False, "error": f"the new text does not fit this file's encoding, so nothing was changed: {e}"}
        check = parse(doc.rel, raw)
        if check.symbols is None or (check.broken and not doc.broken):
            return {"ok": False, "error": f"the result would not parse, so nothing was changed. {check.problem or ''}".rstrip()}
        try:
            self.ws.write_bytes(doc.rel, raw)
        except (OSError, WorkspaceError) as e:
            return {"ok": False, "error": f"it could not be written: {e}"}
        return {"ok": True, "path": doc.rel, "line": sym.line}

    def replace_body(self, symbol: str, text: str) -> dict:
        """Replaces the whole symbol: what `find_symbol(body=True)` shows, signature, decorators and indentation included, so that what was read can be changed and given back."""
        return self._edit(symbol, lambda doc, sym: doc.buf[:sym.start] + doc.fit(text) + doc.buf[sym.stop:])

    def insert_before(self, symbol: str, text: str) -> dict:
        """Puts lines above the symbol, as they are given: the indentation and the blank lines are the caller's."""
        return self._edit(symbol, lambda doc, sym: doc.buf[:sym.start] + doc.fit(text) + doc.eol + doc.buf[sym.start:])

    def insert_after(self, symbol: str, text: str) -> dict:
        """Puts lines below the symbol, as they are given."""
        def change(doc: Doc, sym: Symbol) -> bytes:
            m = _LINE_END.search(doc.buf, sym.stop)
            if m is None:                                                     # the last line of a file with no line end
                return doc.buf + doc.eol + doc.fit(text)
            return doc.buf[:m.end()] + doc.fit(text) + doc.eol + doc.buf[m.end():]
        return self._edit(symbol, change)
