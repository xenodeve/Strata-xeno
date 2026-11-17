"""serve/documents.py - the text of an office document, for the chat and the Read tool (issue #334). Ours.

The chat reads what real work comes in: PDF (serve/pdf_blocks.py: the text layer, a scanned page as an image with
vision), Word .docx, Excel .xlsx, PowerPoint .pptx, OpenDocument .odt/.ods/.odp, EPUB and RTF.  Without this the model
wrote a script, or tried to install a library, to read a file it was given.  Only the standard library (zipfile and
xml.etree) and pypdf, which the server already has: no new dependency.

- The type comes from the file's name, then its first bytes and its zip parts, then the media type (a browser often
  sends none, or application/octet-stream).
- An old binary .doc/.xls/.ppt (an OLE file) says so and how to make it readable; it is not guessed at.
- Limits: MAX_BYTES of file, MAX_UNZIPPED bytes read out of a zip in all (a zip bomb stops there), MAX_ROWS rows per
  sheet and MAX_TEXT characters of text, each with a note where it stops.
- xml.etree never fetches an external entity, and expat refuses an entity expansion bomb.
- A file that cannot be read gives a note, never an exception: the request goes on with the rest.
"""
from __future__ import annotations

import collections
import hashlib
import html.parser
import io
import os
import posixpath
import re
import threading
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile

MAX_BYTES = 30_000_000             # the largest file read
MAX_UNZIPPED = 100_000_000         # bytes read out of one zip, in all its parts (ET parses each part in memory)
MAX_ROWS = 2_000                   # rows per sheet given to the chat
MAX_TEXT = 200_000                 # characters of text given to the chat
MAX_READ_TEXT = 5_000_000          # characters of text the Read tool takes from one file (it reads them by lines)

OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
KINDS = ("pdf", "docx", "xlsx", "pptx", "odt", "ods", "odp", "epub", "rtf", "doc", "xls", "ppt")
LEGACY = {"doc": ".docx", "xls": ".xlsx", "ppt": ".pptx"}
MEDIA = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "application/vnd.oasis.opendocument.text": "odt",
    "application/vnd.oasis.opendocument.spreadsheet": "ods",
    "application/vnd.oasis.opendocument.presentation": "odp",
    "application/epub+zip": "epub",
    "application/rtf": "rtf", "text/rtf": "rtf",
    "application/msword": "doc", "application/vnd.ms-excel": "xls", "application/vnd.ms-powerpoint": "ppt",
}
EXTENSIONS = tuple("." + k for k in KINDS)      # what the Read tool takes to this module


class _Unreadable(Exception):
    """A note for the model: why the file was not read."""


# ------------------------------------------------------------------------------------------------ the type
def kind_of(name: str | None, media_type: str | None, data: bytes) -> str | None:
    """"docx", "xlsx", ... (KINDS), or None for a file this module does not read."""
    ext = os.path.splitext(name or "")[1].lower().lstrip(".")
    if ext in KINDS:
        return ext
    head = data[:8]
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(b"{\\rtf"):
        return "rtf"
    if head.startswith(b"PK"):
        kind = _zip_kind(data)
        if kind:
            return kind
    mt = (media_type or "").split(";")[0].strip().lower()
    if mt in MEDIA:
        return MEDIA[mt]
    if head.startswith(OLE_MAGIC):
        return "doc"
    return None


def _zip_kind(data: bytes) -> str | None:
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
        names = set(z.namelist())
        if "mimetype" in names:
            with z.open("mimetype") as f:
                mt = f.read(100).decode("ascii", "replace").strip()
            if mt in MEDIA:
                return MEDIA[mt]
        for part, kind in (("word/document.xml", "docx"), ("xl/workbook.xml", "xlsx"), ("ppt/presentation.xml", "pptx"),
                           ("META-INF/container.xml", "epub")):
            if part in names:
                return kind
    except Exception:  # noqa: BLE001 - not a zip after all
        return None
    return None


# ------------------------------------------------------------------------------------------------ entry points
CACHE_ENTRIES = 16
_cache: "collections.OrderedDict[tuple, list[dict]]" = collections.OrderedDict()
_cache_lock = threading.Lock()


def extract(name: str, media_type: str | None, data: bytes, vision: bool) -> list[dict]:
    """OpenAI content parts for one file: a text part (and, for a PDF with scanned pages and vision, image parts).
    The web app sends the file again on every turn, so the last CACHE_ENTRIES results are kept by the file's sha256:
    a 300-page PDF took 1.9 s to read each turn without it."""
    key = (hashlib.sha256(data).hexdigest(), name, media_type, bool(vision), MAX_BYTES, MAX_TEXT, MAX_ROWS, MAX_UNZIPPED)
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
            return [dict(p) for p in hit]
    parts = _extract(name, media_type, data, vision)
    with _cache_lock:
        _cache[key] = parts
        while len(_cache) > CACHE_ENTRIES:
            _cache.popitem(last=False)
    return [dict(p) for p in parts]


def _extract(name: str, media_type: str | None, data: bytes, vision: bool) -> list[dict]:
    name = name or "document"
    if len(data) > MAX_BYTES:
        return [_note(name, f"(the file is {len(data):,} bytes, more than the {MAX_BYTES:,} that can be read)")]
    kind = kind_of(name, media_type, data)
    if kind == "pdf":
        from serve import pdf_blocks
        return _capped(pdf_blocks.pdf_parts(data, name, vision, max_text=MAX_TEXT), name)
    body, ok = text_of(name, data, media_type=media_type, limit=MAX_TEXT)
    return [_note(name, body if ok else f"({body})")]


def text_of(name: str, data: bytes, max_rows: int | None = None, media_type: str | None = None,
            limit: int | None = None) -> tuple[str, bool]:
    """(the file's text, True), or (why it cannot be read, False).  `limit`: the characters to give at most (default
    MAX_READ_TEXT, the Read tool's); every reader stops while it builds the text once the limit is passed (a 469-byte
    .ods of repeated cells made 787 MB before, review of #334), and a note says where it was cut."""
    limit = MAX_READ_TEXT if limit is None else limit
    kind = kind_of(name, media_type, data)
    sink = _Sink(limit)
    try:
        if kind in LEGACY:
            if data.lstrip().startswith(b"{\\rtf"):
                kind = "rtf"                                       # Word saves RTF (and HTML) under an old name, too
            elif data.startswith(b"PK"):
                kind = _zip_kind(data)                             # a new file with an old name
            elif data.startswith(OLE_MAGIC) or b"\0" in data[:8192]:
                return (f"{name} is in the old binary Office format (.{kind}), which cannot be read here. Save it as "
                        f"{LEGACY[kind]} (File > Save As) and give it again"), False
            else:
                kind = None                                        # text (an HTML export): read as text below
        elif kind and data.startswith(OLE_MAGIC):
            return (f"{name} is protected by a password (or is an old binary Office file with a new name), so it "
                    "cannot be read. Save it again without a password, as .docx, .xlsx or .pptx"), False
        reader = READERS.get(kind or "")
        if reader is None:
            if b"\0" in data[:8192]:
                return f"{name}: this kind of file cannot be read here (it reads PDF, Word, Excel, PowerPoint, OpenDocument, EPUB and RTF)", False
            try:
                body = data.decode("utf-8-sig")
            except UnicodeDecodeError:
                return f"{name}: this kind of file cannot be read here (it reads PDF, Word, Excel, PowerPoint, OpenDocument, EPUB and RTF)", False
            reader = lambda _data, s, _rows: s.add_text(body)  # noqa: E731
        try:
            reader(data, sink, max_rows or MAX_ROWS)
        except _Full:
            pass
        body = _tidy(sink.text())
        if sink.full:
            body += f"\n[cut here at {limit:,} characters: the rest of {name} is not shown]"
        return body, True
    except _Unreadable as e:
        return f"{name} could not be read: {e}", False
    except Exception as e:  # noqa: BLE001 - a file that cannot be parsed says so; the request goes on
        return f"{name} could not be read ({type(e).__name__}); the file may be damaged", False


def _note(name: str, body: str) -> dict:
    return {"type": "text", "text": f"[Document: {name}]\n{body}"}


def _capped(parts: list[dict], name: str) -> list[dict]:
    """A PDF's parts with at most MAX_TEXT characters of page text (the "[Document: ...]" line is not counted);
    what is past it is left out, with a note."""
    out, n = parts[:1], 0
    for p in parts[1:]:
        if p.get("type") == "text":
            t = p["text"]
            if n + len(t) > MAX_TEXT:
                out.append({"type": "text", "text": t[:max(0, MAX_TEXT - n)]
                            + f"\n[cut here at {MAX_TEXT:,} characters: the rest of {name} is not shown]"})
                return out
            n += len(t)
        out.append(p)
    return out


def _tidy(s: str) -> str:
    """Trailing spaces off each line and no run of more than one empty line."""
    s = "\n".join(line.rstrip() for line in s.replace("\r\n", "\n").replace("\r", "\n").split("\n"))
    return re.sub(r"\n{3,}", "\n\n", s).strip("\n")


class _Full(Exception):
    """The text reached its limit: the reader stops here."""


class _Sink:
    """The text a reader writes, line by line, up to `limit` characters: past it, `add` keeps what fits and raises
    _Full, so a reader never builds more than the limit (and one line never more than what is left of it)."""

    def __init__(self, limit: int):
        self.limit, self.n, self.lines, self.full = limit, 0, [], False

    def left(self) -> int:
        return max(0, self.limit - self.n)

    def add(self, line: str):
        if not line and self.lines and not self.lines[-1]:
            return                                                      # runs of empty lines are one
        if self.n + len(line) + 1 > self.limit:
            self.lines.append(line[:self.left()])
            self.n, self.full = self.limit, True
            raise _Full
        self.lines.append(line)
        self.n += len(line) + 1

    def add_text(self, text: str):
        for line in text.split("\n"):
            self.add(line)

    def text(self) -> str:
        return "\n".join(self.lines)


def _pdf_text(data: bytes, sink: _Sink, _rows: int):
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    for i, p in enumerate(reader.pages, 1):
        sink.add(f"--- page {i} ---")
        sink.add_text((p.extract_text() or "").strip())
        sink.add("")


# ------------------------------------------------------------------------------------------------ zip and XML
PART_MAX = 20_000_000              # bytes of one part parsed whole (ElementTree takes ~15x that in memory)


class _Zip:
    """A zip read within one byte budget for all its parts: a part's stated size is not trusted."""

    def __init__(self, data: bytes):
        try:
            self.z = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile:
            raise _Unreadable("it is not a valid zip package (the file may be damaged)") from None
        self.left = MAX_UNZIPPED
        self.names = self.z.namelist()

    def read(self, name: str, need: bool = True) -> bytes | None:
        """A whole part (for one parsed whole, or HTML): at most PART_MAX bytes."""
        try:
            info = self.z.getinfo(name)
        except KeyError:
            if need:
                raise _Unreadable(f"its part {name} is missing") from None
            return None
        with self.z.open(info) as f:
            out = f.read(min(self.left, PART_MAX) + 1)
        if len(out) > PART_MAX:
            raise _Unreadable(f"its part {name} unpacks to more than {PART_MAX:,} bytes, which is too large to read")
        if len(out) > self.left:
            raise _Unreadable(f"it unpacks to more than {MAX_UNZIPPED:,} bytes, which is too large to read")
        self.left -= len(out)
        return out

    def stream(self, name: str, depth: int | None = None, want: set | None = None):
        """The elements of a part, each given when it ends and then dropped from the tree, so a large part is never
        held whole: those at `depth` (1 is the root's children), or those named in `want`."""
        if name not in self.names:
            raise _Unreadable(f"its part {name} is missing")
        stack: list[ET.Element] = []
        with _Budgeted(self, self.z.open(name)) as raw:
            for ev, el in ET.iterparse(io.BufferedReader(raw), events=("start", "end")):
                if ev == "start":
                    stack.append(el)
                    continue
                stack.pop()
                if (depth is not None and len(stack) == depth) or (want is not None and _local(el.tag) in want):
                    yield el
                    if stack:
                        stack[-1].remove(el)


class _Budgeted(io.RawIOBase):
    def __init__(self, owner: _Zip, f):
        self.owner, self.f = owner, f

    def readable(self):
        return True

    def readinto(self, b):
        n = self.f.readinto(b)
        self.owner.left -= n
        if self.owner.left < 0:
            raise _Unreadable(f"it unpacks to more than {MAX_UNZIPPED:,} bytes, which is too large to read")
        return n

    def close(self):
        self.f.close()
        super().close()


def _xml(data: bytes) -> ET.Element:
    try:
        return ET.fromstring(data)
    except ET.ParseError as e:
        raise _Unreadable(f"its XML is damaged ({e})") from None


def _local(tag) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _attr(el: ET.Element, name: str, default=None):
    for k, v in el.attrib.items():
        if k == name or k.endswith("}" + name):
            return v
    return default


def _rid(el: ET.Element) -> str:
    """An element's relationship id (r:id), not a plain id attribute beside it (a slide's id="256")."""
    return next((v for k, v in el.attrib.items() if k.endswith("}id")), "")


def _rels(z: _Zip, part: str) -> dict[str, str]:
    """A part's relationships: id -> the target's path in the zip."""
    folder, file = posixpath.split(part)
    raw = z.read(posixpath.join(folder, "_rels", file + ".rels"), need=False)
    out = {}
    for r in (_xml(raw) if raw else []):
        target = r.get("Target") or ""
        out[r.get("Id")] = target.lstrip("/") if target.startswith("/") else posixpath.normpath(posixpath.join(folder, target))
    return out


# ------------------------------------------------------------------------------------------------ Word
DOCX_SKIP = {"pPr", "rPr", "delText", "instrText", "fldData", "Fallback"}   # formatting (a tab stop is not a tab), deleted text,
                                                                           # field codes, the old-reader copy of a drawing


def _docx(data: bytes, sink: _Sink, _rows: int):
    for el in _Zip(data).stream("word/document.xml", depth=2):          # the body's children, one at a time
        _docx_el(el, sink.add)


def _docx_block(el: ET.Element, emit):
    for c in el:
        _docx_el(c, emit)


def _docx_el(c: ET.Element, emit):
    n = _local(c.tag)
    if n == "p":
        text, boxes = _docx_para(c)
        emit(text)
        for line in boxes:
            emit(line)
    elif n == "tbl":
        for tr in _docx_children(c, "tr"):
            cells = []
            for tc in _docx_children(tr, "tc"):
                lines: list[str] = []
                _docx_block(tc, lines.append)
                cells.append(" ".join(x.strip() for x in lines if x.strip()))
            emit(" | ".join(cells))
        emit("")
    elif n not in ("sectPr", "tblPr", "tblGrid", "trPr", "tcPr"):
        _docx_block(c, emit)                                            # a content control, a custom XML wrapper, ...


def _docx_children(el: ET.Element, name: str):
    """The rows of a table (or cells of a row), also inside a content control, but not those of a table inside it."""
    for c in el:
        n = _local(c.tag)
        if n == name:
            yield c
        elif n not in ("tbl", "tc", "tr"):
            yield from _docx_children(c, name)


def _docx_para(p: ET.Element) -> tuple[str, list[str]]:
    """A paragraph's text, and the lines of the text boxes it holds (each read once: the old-reader Fallback is skipped)."""
    s: list[str] = []
    boxes: list[str] = []

    def walk(el):
        for e in el:
            n = _local(e.tag)
            if n in DOCX_SKIP:
                continue
            if n == "t":
                s.append(e.text or "")
            elif n == "tab":
                s.append("\t")
            elif n in ("br", "cr"):
                s.append("\n")
            elif n == "txbxContent":
                _docx_block(e, boxes.append)
            else:
                walk(e)
    walk(p)
    return "".join(s), boxes


# ------------------------------------------------------------------------------------------------ Excel
MAX_COLS = 256            # columns written in place; a cell past them is written with its reference ("XFD7: value")


def _col(ref: str) -> int:
    n = 0
    for ch in ref:
        if not ch.isalpha():
            break
        n = n * 26 + (ord(ch.upper()) - 64)
    return n - 1


def _col_name(i: int) -> str:
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def _si_text(si: ET.Element) -> str:
    """A shared or inline string: its text, or its rich-text runs joined; a phonetic run (rPh) is left out."""
    out = []
    for c in si:
        n = _local(c.tag)
        if n == "t":
            out.append(c.text or "")
        elif n == "r":
            out += [t.text or "" for t in c if _local(t.tag) == "t"]
    return "".join(out)


def _xlsx(data: bytes, sink: _Sink, max_rows: int):
    z = _Zip(data)
    wb = _xml(z.read("xl/workbook.xml"))
    rels = _rels(z, "xl/workbook.xml")
    shared = []
    if "xl/sharedStrings.xml" in z.names:
        shared = [_si_text(si) for si in z.stream("xl/sharedStrings.xml", want={"si"})]
    sheets = [s for s in wb.iter() if _local(s.tag) == "sheet"]
    for i, s in enumerate(sheets, 1):
        path = rels.get(_rid(s), f"xl/worksheets/sheet{i}.xml")
        sink.add(f"=== sheet: {s.get('name') or i} ===")
        if path not in z.names:
            sink.add("(this sheet's part is missing)")
            continue
        _xlsx_rows(z, path, shared, max_rows, sink)
        sink.add("")


def _xlsx_rows(z: _Zip, path: str, shared: list[str], max_rows: int, sink: _Sink):
    rows = 0
    for el in z.stream(path, want={"row"}):
        cells: dict[int, str] = {}
        nxt = 0
        for c in el:
            if _local(c.tag) != "c":
                continue
            col = _col(c.get("r") or "") if c.get("r") else nxt
            nxt = col + 1
            t, v = c.get("t"), next((x.text or "" for x in c if _local(x.tag) == "v"), None)
            if t == "s" and v is not None:
                try:
                    val = shared[int(v)]
                except (ValueError, IndexError):
                    val = ""
            elif t == "inlineStr":
                val = "".join(_si_text(x) for x in c if _local(x.tag) == "is")
            elif t == "b":
                val = "TRUE" if v == "1" else "FALSE"
            else:
                val = v or ""
            if val != "" and 0 <= col < 16384:
                cells[col] = val.replace("\t", " ").replace("\n", " ")
        if not cells:
            continue
        if rows >= max_rows:
            sink.add(f"[only the first {max_rows:,} rows of this sheet are shown]")
            return
        rows += 1
        near = [k for k in cells if k < MAX_COLS]
        line = [cells.get(j, "") for j in range(max(near) + 1)] if near else []
        rownum = el.get("r") or str(rows)
        line += [f"{_col_name(k)}{rownum}: {cells[k]}" for k in sorted(cells) if k >= MAX_COLS]
        sink.add("\t".join(line))


# ------------------------------------------------------------------------------------------------ PowerPoint
def _pptx(data: bytes, sink: _Sink, _rows: int):
    z = _Zip(data)
    order: list[str] = []
    if "ppt/presentation.xml" in z.names:                               # the order the deck shows them in
        rels = _rels(z, "ppt/presentation.xml")
        pres = _xml(z.read("ppt/presentation.xml"))
        order = [rels.get(_rid(s)) for s in pres.iter() if _local(s.tag) == "sldId"]
        order = [p for p in order if p in z.names]
    if not order:                                                       # no list: by the number in the file name
        found = [(re.fullmatch(r"ppt/slides/slide(\d+)\.xml", n), n) for n in z.names]
        order = [n for _, n in sorted((int(m.group(1)), n) for m, n in found if m)]
    for num, name in enumerate(order, 1):
        root = _xml(z.read(name))
        sink.add(f"--- slide {num} ---")
        for p in root.iter():
            if _local(p.tag) == "p":
                line = "".join(t.text or "" for t in p.iter() if _local(t.tag) == "t")
                if line.strip():
                    sink.add(line)
        sink.add("")


# ------------------------------------------------------------------------------------------------ OpenDocument
REPEAT_CAP = 256          # a repeated cell or row that has text is written at most this many times
SPACE_CAP = 100           # the spaces one <text:s text:c="..."/> gives at most


def _odf(data: bytes, sink: _Sink, max_rows: int = MAX_ROWS):
    root = _xml(_Zip(data).read("content.xml"))
    _odf_block(root, sink, max_rows, {"slide": 0})


def _odf_block(el: ET.Element, sink: _Sink, max_rows: int, state: dict):
    for c in el:
        n = _local(c.tag)
        if n in ("p", "h"):
            sink.add(_odf_text(c))
        elif n == "table":
            sink.add(f"=== sheet: {_attr(c, 'name') or 'table'} ===")
            _odf_table(c, max_rows, sink)
            sink.add("")
        elif n == "page":
            state["slide"] += 1
            sink.add(f"--- slide {state['slide']} ---")
            _odf_block(c, sink, max_rows, state)
            sink.add("")
        elif n not in ("automatic-styles", "font-face-decls", "scripts", "forms", "notes", "annotation"):
            _odf_block(c, sink, max_rows, state)


def _odf_text(el: ET.Element) -> str:
    s = [el.text or ""]
    for c in el:
        n = _local(c.tag)
        if n == "s":
            try:
                s.append(" " * max(1, min(SPACE_CAP, int(_attr(c, "c") or 1))))
            except ValueError:
                s.append(" ")
        elif n == "tab":
            s.append("\t")
        elif n == "line-break":
            s.append("\n")
        elif n not in ("annotation", "note-citation"):
            s.append(_odf_text(c))
        s.append(c.tail or "")
    return "".join(s)


def _repeat(el: ET.Element, name: str) -> int:
    try:
        return max(1, int(_attr(el, name) or 1))
    except ValueError:
        return 1


def _odf_table(table: ET.Element, max_rows: int, sink: _Sink):
    rows = 0
    for tr in table.iter():
        if _local(tr.tag) != "table-row":
            continue
        cells: list[str] = []
        size, room = 0, sink.left() + 1                                 # a line never grows past what the text has left
        empties = 0
        for tc in tr:
            if _local(tc.tag) not in ("table-cell", "covered-table-cell"):
                continue
            val = " ".join(_odf_text(p) for p in tc if _local(p.tag) in ("p", "h")).replace("\t", " ").replace("\n", " ")
            rep = _repeat(tc, "number-columns-repeated")
            if not val:
                empties += rep                                          # an empty run is written only if text follows it
                continue
            if empties:
                pad = min(empties, 16384 - len(cells), room - size)
                cells += [""] * max(0, pad)
                size += max(0, pad)
                empties = 0
            for _ in range(min(rep, REPEAT_CAP)):
                if size > room:
                    break
                cells.append(val)
                size += len(val) + 1
            if size > room:
                break
        if not cells:
            continue
        line = "\t".join(cells)
        for _ in range(min(_repeat(tr, "number-rows-repeated"), REPEAT_CAP)):
            if rows >= max_rows:
                sink.add(f"[only the first {max_rows:,} rows of this sheet are shown]")
                return
            rows += 1
            sink.add(line)



# ------------------------------------------------------------------------------------------------ EPUB
class _HtmlText(html.parser.HTMLParser):
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "section", "article",
             "table", "ul", "ol", "dt", "dd", "hr", "figcaption"}
    SKIP = {"script", "style", "head", "title"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in self.BLOCK:
            self.out.append("\n")
        elif tag in ("td", "th"):
            self.out.append(" | ")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def _html_text(raw: bytes) -> str:
    p = _HtmlText()
    p.feed(raw.decode("utf-8", "replace"))
    p.close()
    return re.sub(r"[ \t]*\n[ \t]*", "\n", "".join(p.out))


def _epub(data: bytes, sink: "_Sink", _rows: int):
    z = _Zip(data)
    container = _xml(z.read("META-INF/container.xml"))
    rootfile = next((r.get("full-path") for r in container.iter() if _local(r.tag) == "rootfile"), None)
    if not rootfile:
        raise _Unreadable("its container names no package file")
    opf = _xml(z.read(rootfile))
    base = posixpath.dirname(rootfile)
    items = {i.get("id"): i.get("href") or "" for i in opf.iter() if _local(i.tag) == "item"}
    read = 0
    for ref in (r for r in opf.iter() if _local(r.tag) == "itemref"):
        href = items.get(ref.get("idref"))
        if not href:
            continue
        path = posixpath.normpath(posixpath.join(base, urllib.parse.unquote(href.split("#")[0])))
        raw = z.read(path, need=False)
        if raw is not None:
            read += 1
            sink.add_text(_html_text(raw))
            sink.add("")
    if not read:
        raise _Unreadable("its reading order names no chapter that is in the file")


# ------------------------------------------------------------------------------------------------ RTF
RTF_TOKEN = re.compile(rb"\\([a-zA-Z]{1,32})(-?\d{1,10})? ?|\\'([0-9a-fA-F]{2})|\\([^a-zA-Z])|([{}])|[\r\n]+|([^\\{}\r\n]+)")
RTF_SKIP = {"fonttbl", "colortbl", "stylesheet", "info", "pict", "object", "themedata", "colorschememapping",
            "datastore", "latentstyles", "listtable", "listoverridetable", "rsidtbl", "generator", "xmlnstbl",
            "mmathPr", "fldinst", "header", "headerl", "headerr", "headerf", "footer", "footerl", "footerr", "footerf",
            "filetbl", "revtbl", "pgdsctbl", "bkmkstart", "bkmkend", "xe", "tc", "nonshppict", "fchars", "lchars"}
RTF_CHARS = {"par": "\n", "line": "\n", "sect": "\n", "page": "\n", "row": "\n", "cell": " | ", "tab": "\t",
             "emdash": "\u2014", "endash": "\u2013", "bullet": "\u2022", "lquote": "\u2018", "rquote": "\u2019",
             "ldblquote": "\u201c", "rdblquote": "\u201d", "emspace": " ", "enspace": " ", "qmspace": " "}


def _rtf(data: bytes) -> str:
    if not data.lstrip().startswith(b"{\\rtf"):
        raise _Unreadable("it does not start as an RTF file")
    m = re.search(rb"\\ansicpg(\d+)", data[:2000])
    codepage = f"cp{int(m.group(1))}" if m else "cp1252"
    try:
        "".encode(codepage)
    except LookupError:
        codepage = "cp1252"
    out: list[str] = []
    pending = bytearray()                                    # \'hh bytes, decoded together (a double-byte code page)
    stack: list[tuple[bool, int]] = []
    skip, uc, to_skip = False, 1, 0

    def flush():
        if pending:
            out.append(pending.decode(codepage, "replace"))
            pending.clear()

    for m in RTF_TOKEN.finditer(data):
        word, arg, hexa, sym, brace, plain = m.groups()
        if brace is not None:
            flush()
            to_skip = 0
            if brace == b"{":
                stack.append((skip, uc))
            elif stack:
                skip, uc = stack.pop()
            continue
        if word is not None:
            w = word.decode("ascii")
            flush()
            if to_skip and w != "u":
                to_skip = 0
            if w in RTF_SKIP:
                skip = True
            elif w == "uc":
                uc = int(arg or 1)
            elif skip:
                pass
            elif w == "u":
                n = int(arg or 0)
                n = n + 65536 if -32768 <= n < 0 else n
                if 0 <= n <= 0x10FFFF:                              # out of range: left out, not the whole file lost
                    out.append(chr(n))
                to_skip = uc
            elif w in RTF_CHARS:
                out.append(RTF_CHARS[w])
            continue
        if hexa is not None:
            if to_skip:
                to_skip -= 1
            elif not skip:
                pending.append(int(hexa, 16))
            continue
        flush()
        if sym is not None:
            if sym == b"*":
                skip = True
            elif sym in (b"\\", b"{", b"}") and not skip:
                out.append(sym.decode())
            elif sym == b"~" and not skip:
                out.append("\u00a0")
            elif sym == b"_" and not skip:
                out.append("-")
            elif sym in (b"\r", b"\n") and not skip:
                out.append("\n")
            continue
        if plain is not None and not skip:
            s = plain.decode(codepage, "replace")
            if to_skip:
                cut = min(to_skip, len(s))
                s, to_skip = s[cut:], to_skip - cut
            out.append(s)
    flush()
    # a character past U+FFFF comes as two \u surrogates: join them, and a lone one becomes U+FFFD
    return "".join(out).encode("utf-16", "surrogatepass").decode("utf-16", "replace")


READERS = {"docx": _docx, "xlsx": _xlsx, "pptx": _pptx, "odt": _odf, "ods": _odf, "odp": _odf, "epub": _epub,
           "rtf": lambda data, sink, _rows: sink.add_text(_rtf(data)), "pdf": _pdf_text}
