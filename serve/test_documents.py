"""The chat reads office documents (issue #334, serve/documents.py): PDF through serve/pdf_blocks.py, and .docx, .xlsx,
.pptx, .odt/.ods/.odp, .epub and .rtf with the standard library.  A legacy binary .doc/.xls/.ppt, a file that cannot be
parsed and an unknown type each give a note, never an exception.  The samples are written here by hand: the minimal
parts each format needs, as a real office program writes them."""
from __future__ import annotations

import base64

import io
import sys
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from serve import documents, pdf_blocks  # noqa: E402
from serve.test_media import text_pdf  # noqa: E402

try:
    import pypdf  # noqa: F401
except ImportError:                                  # pragma: no cover
    pypdf = None

W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
S = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
R = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
A = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
P = 'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"'
ODF_NS = ('xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
          'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
          'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" '
          'xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"')


def zipped(files: dict[str, str | bytes], stored_first: str | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        if stored_first:
            z.writestr(zipfile.ZipInfo(stored_first), files[stored_first], compress_type=zipfile.ZIP_STORED)
        for name, body in files.items():
            if name != stored_first:
                z.writestr(name, body)
    return buf.getvalue()


def docx(body: str, types: bool = True) -> bytes:
    files = {"word/document.xml": f'<?xml version="1.0" encoding="UTF-8"?><w:document {W}><w:body>{body}</w:body></w:document>'}
    if types:
        files["[Content_Types].xml"] = ('<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                                        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-'
                                        'officedocument.wordprocessingml.document.main+xml"/></Types>')
    return zipped(files)


def para(*runs: str) -> str:
    return "<w:p>" + "".join(f"<w:r><w:t xml:space=\"preserve\">{r}</w:t></w:r>" for r in runs) + "</w:p>"


def xlsx(sheets: list[tuple[str, str]], shared: list[str]) -> bytes:
    """sheets: (name, the <sheetData> rows), in the workbook's order; written as sheet files in the reverse order, so
    the reader has to take the order from workbook.xml."""
    n = len(sheets)
    files = {
        "[Content_Types].xml": '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>',
        "xl/workbook.xml": f'<workbook {S} {R}><sheets>' + "".join(
            f'<sheet name="{name}" sheetId="{i + 1}" r:id="rId{i + 1}"/>' for i, (name, _) in enumerate(sheets)) + "</sheets></workbook>",
        "xl/_rels/workbook.xml.rels": '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + "".join(
            f'<Relationship Id="rId{i + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="{"/xl/" if i == 0 else ""}worksheets/sheet{n - i}.xml"/>' for i in range(n)) + "</Relationships>",
        "xl/sharedStrings.xml": f'<sst {S}>' + "".join(f"<si><t>{s}</t></si>" for s in shared)
        + '<si><r><t>rich </t></r><r><rPr><b/></rPr><t>text</t></r><rPh><t>PHONETIC</t></rPh></si></sst>',
    }
    for i, (_, rows) in enumerate(sheets):
        files[f"xl/worksheets/sheet{n - i}.xml"] = f'<worksheet {S}><sheetData>{rows}</sheetData></worksheet>'
    return zipped(files)


def pptx(slides: dict[int, list[str]]) -> bytes:
    files = {"[Content_Types].xml": '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>',
             "ppt/presentation.xml": f"<p:presentation {P}/>"}
    for n, paras in slides.items():
        body = "".join(f"<a:p><a:r><a:t>{t}</a:t></a:r></a:p>" for t in paras)
        files[f"ppt/slides/slide{n}.xml"] = (f"<p:sld {P} {A}><p:cSld><p:spTree><p:sp><p:txBody>{body}</p:txBody></p:sp>"
                                             "</p:spTree></p:cSld></p:sld>")
    return zipped(files)


def odf(kind: str, body: str) -> bytes:
    return zipped({"mimetype": f"application/vnd.oasis.opendocument.{kind}",
                   "content.xml": f'<office:document-content {ODF_NS}><office:body>{body}</office:body></office:document-content>'},
                  stored_first="mimetype")


def epub(chapters: list[tuple[str, str]], spine: list[str]) -> bytes:
    files = {"mimetype": "application/epub+zip",
             "META-INF/container.xml": ('<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
                                        '<rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
                                        "</rootfiles></container>"),
             "OEBPS/content.opf": ('<package xmlns="http://www.idpf.org/2007/opf"><manifest>' + "".join(
                 f'<item id="{href.split(".")[0].replace("/", "-").replace("%20", "")}" href="{href}" media-type="application/xhtml+xml"/>'
                 for href, _ in chapters) + "</manifest><spine>" + "".join(f'<itemref idref="{i}"/>' for i in spine)
                 + "</spine></package>")}
    for href, html in chapters:
        files["OEBPS/" + href.replace("%20", " ")] = html
    return zipped(files, stored_first="mimetype")


def text(parts: list[dict]) -> str:
    return "\n".join(p["text"] for p in parts if p.get("type") == "text")


class PdfStaysAsItWas(unittest.TestCase):
    """document_parts' PDF branch is byte-identical to before #334 (values captured from the code before the change)."""

    def block(self, data: bytes, title="r.pdf", media="application/pdf"):
        return {"title": title, "source": {"type": "base64", "media_type": media, "data": base64.b64encode(data).decode()}}

    @unittest.skipIf(pypdf is None, "needs pypdf")
    def test_text_pages_and_a_scan_without_vision(self):
        self.assertEqual(pdf_blocks.document_parts(self.block(text_pdf(["first page", None, "third"])), False), [
            {"type": "text", "text": "[Document: r.pdf, 3 page(s)]"},
            {"type": "text", "text": "--- page 1 ---\nfirst page"},
            {"type": "text", "text": "--- page 2 --- (no text layer; images are not enabled on this server)"},
            {"type": "text", "text": "--- page 3 ---\nthird"}])

    @unittest.skipIf(pypdf is None, "needs pypdf")
    def test_a_scan_with_vision_is_the_same_image(self):
        data = text_pdf(["first page", None, "third"])
        v = pdf_blocks.document_parts(self.block(data), True)
        self.assertEqual([p["type"] for p in v], ["text", "text", "text", "image_url", "text"])
        self.assertEqual(v[2]["text"], "--- page 2 (scanned, as an image) ---")
        # the page as the renderer draws it (sha256 0806f031... before the change, with pypdfium2 and Pillow of 2026-11-17)
        self.assertEqual(v[3]["image_url"]["url"], pdf_blocks._page_image(data, 1))

    @unittest.skipIf(pypdf is None, "needs pypdf")
    def test_the_notes_are_the_same(self):
        bad = {"source": {"type": "base64", "media_type": "application/pdf", "data": base64.b64encode(b"%PDF-1.4 junk").decode()}}
        self.assertEqual(pdf_blocks.document_parts(bad), [{"type": "text", "text": "[Document: document]\n(the PDF could not be read: PdfStreamError)"}])
        self.assertEqual(pdf_blocks.document_parts({"title": "x", "source": {"type": "url", "url": "http://a"}}),
                         [{"type": "text", "text": "[Document: x]\n(unsupported document source: url)"}])
        self.assertEqual(pdf_blocks.document_parts({"title": "x", "source": {"type": "text", "data": "plain"}}),
                         [{"type": "text", "text": "[Document: x]\nplain"}])


class ReadOnceForEveryTurn(unittest.TestCase):
    """The web app sends a file again on every turn: it is read once, and a long PDF only up to the text limit."""

    def test_the_same_file_is_read_once(self):
        calls = []
        real = documents._extract
        documents._extract = lambda *a: calls.append(a) or real(*a)
        self.addCleanup(setattr, documents, "_extract", real)
        data = docx(para("cached ", str(id(self))))
        first = documents.extract("c.docx", None, data, False)
        first[0]["text"] = "changed by a caller"
        again = documents.extract("c.docx", None, data, False)
        self.assertEqual(len(calls), 1)
        self.assertIn("cached", again[0]["text"])
        documents.extract("c.docx", None, data, True)                       # another vision setting is another entry
        self.assertEqual(len(calls), 2)

    @unittest.skipIf(pypdf is None, "needs pypdf")
    def test_a_long_pdf_stops_reading_at_the_text_limit(self):
        old = documents.MAX_TEXT
        documents.MAX_TEXT = 30
        self.addCleanup(setattr, documents, "MAX_TEXT", old)
        seen = []
        real = pdf_blocks.pdf_parts
        pdf_blocks.pdf_parts = lambda *a, **k: seen.append(k) or real(*a, **k)
        self.addCleanup(setattr, pdf_blocks, "pdf_parts", real)
        t = text(documents.extract("long.pdf", None, text_pdf([f"page text number {i}" for i in range(1, 9)]), False))
        self.assertEqual(seen, [{"max_text": 30}])
        self.assertIn("[Document: long.pdf, 8 page(s)]", t)              # the count is of all the pages
        self.assertIn("[cut here", t)
        self.assertNotIn("number 3", t)
        whole = pdf_blocks.pdf_parts(text_pdf([f"p{i}" for i in range(1, 4)]), "w.pdf", False)
        self.assertEqual(len(whole), 4)


class OfficeThroughDocumentParts(unittest.TestCase):
    def test_a_docx_block_is_read(self):
        data = docx(para("สวัสดี ", "Word"))
        block = {"title": "a.docx", "source": {"type": "base64", "data": base64.b64encode(data).decode(),
                                               "media_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}}
        self.assertIn("สวัสดี Word", text(pdf_blocks.document_parts(block, False)))


class Word(unittest.TestCase):
    def test_paragraphs_in_order_and_tables_as_rows(self):
        body = (para("First ", "paragraph") + "<w:p><w:r><w:t>tab</w:t><w:tab/><w:t>bed</w:t><w:br/><w:t>line two</w:t></w:r></w:p>"
                + "<w:tbl><w:tr><w:tc>" + para("ชื่อ") + "</w:tc><w:tc>" + para("Age") + "</w:tc></w:tr>"
                + "<w:tr><w:tc>" + para("Ann") + "</w:tc><w:tc>" + para("31") + "</w:tc></w:tr></w:tbl>"
                + "<w:sdt><w:sdtContent>" + para("in a content control") + "</w:sdtContent></w:sdt>" + para("Last"))
        t = text(documents.extract("r.docx", None, docx(body), False))
        self.assertIn("[Document: r.docx]", t)
        self.assertIn("First paragraph", t)
        self.assertIn("tab\tbed\nline two", t)
        self.assertIn("ชื่อ | Age\nAnn | 31", t)
        self.assertLess(t.index("First paragraph"), t.index("ชื่อ | Age"))
        self.assertLess(t.index("in a content control"), t.index("Last"))


class Excel(unittest.TestCase):
    def test_every_sheet_by_name_in_the_workbook_order_with_strings_resolved(self):
        rows1 = ('<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>'
                 '<row r="2"><c r="A2"><v>42</v></c><c r="C2" t="inlineStr"><is><t>inline ไทย</t></is></c></row>'
                 '<row r="3"><c r="B3" t="s"><v>2</v></c><c r="C3" t="b"><v>1</v></c></row>')
        rows2 = '<row r="1"><c r="A1" t="str"><v>formula result</v></c></row>'
        t = text(documents.extract("book.xlsx", None, xlsx([("ยอดขาย", rows1), ("Second", rows2)], ["Name", "ราคา"]), False))
        self.assertIn("Name\tราคา", t)
        self.assertIn("42\t\tinline ไทย", t)                 # B2 is empty: the cell keeps its column
        self.assertIn("\trich text\tTRUE", t)                # rich runs joined, the phonetic run left out
        self.assertNotIn("PHONETIC", t)
        self.assertLess(t.index("ยอดขาย"), t.index("Name"))
        self.assertLess(t.index("Name"), t.index("Second"))
        self.assertLess(t.index("Second"), t.index("formula result"))

    def test_a_long_sheet_stops_at_the_row_limit_and_says_so(self):
        rows = "".join(f'<row r="{i}"><c r="A{i}"><v>{i}</v></c></row>' for i in range(1, 51))
        old = documents.MAX_ROWS
        documents.MAX_ROWS = 10
        self.addCleanup(setattr, documents, "MAX_ROWS", old)
        t = text(documents.extract("big.xlsx", None, xlsx([("S", rows)], []), False))
        self.assertIn("\n10\n", t + "\n")
        self.assertNotIn("\n11\n", t + "\n")
        self.assertIn("only the first 10 rows", t)


class PowerPoint(unittest.TestCase):
    def test_slides_in_numeric_order(self):
        t = text(documents.extract("deck.pptx", None, pptx({10: ["ten"], 2: ["two", "สอง"], 1: ["one"]}), False))
        self.assertLess(t.index("--- slide 1 ---"), t.index("one"))
        self.assertLess(t.index("one"), t.index("--- slide 2 ---"))
        self.assertIn("two\nสอง", t)
        self.assertLess(t.index("two"), t.index("--- slide 10 ---"))


class OpenDocument(unittest.TestCase):
    def test_odt_text_with_spaces_tabs_and_lists(self):
        body = ('<office:text><text:h>Heading</text:h><text:p>a<text:s text:c="3"/>b<text:tab/>c<text:line-break/>d</text:p>'
                '<text:list><text:list-item><text:p>item ไทย</text:p></text:list-item></text:list></office:text>')
        t = text(documents.extract("n.odt", None, odf("text", body), False))
        self.assertIn("Heading", t)
        self.assertIn("a   b\tc\nd", t)
        self.assertIn("item ไทย", t)

    def test_ods_rows_without_expanding_huge_empty_repeats(self):
        body = ('<office:spreadsheet><table:table table:name="Sheet ก">'
                '<table:table-row><table:table-cell><text:p>x</text:p></table:table-cell>'
                '<table:table-cell table:number-columns-repeated="2"><text:p>y</text:p></table:table-cell>'
                '<table:table-cell table:number-columns-repeated="16380"/></table:table-row>'
                '<table:table-row table:number-rows-repeated="1048570"><table:table-cell table:number-columns-repeated="16384"/></table:table-row>'
                '<table:table-row><table:table-cell table:number-columns-repeated="2"/><table:table-cell><text:p>z</text:p></table:table-cell></table:table-row>'
                "</table:table></office:spreadsheet>")
        t = text(documents.extract("s.ods", None, odf("spreadsheet", body), False))
        self.assertIn("Sheet ก", t)
        self.assertIn("x\ty\ty", t)
        self.assertIn("\t\tz", t)
        self.assertLess(len(t), 500)

    def test_odp_by_page(self):
        body = ('<office:presentation><draw:page draw:name="p1"><draw:frame><draw:text-box><text:p>first slide</text:p>'
                '</draw:text-box></draw:frame></draw:page><draw:page><draw:frame><draw:text-box><text:p>second</text:p>'
                "</draw:text-box></draw:frame></draw:page></office:presentation>")
        t = text(documents.extract("p.odp", None, odf("presentation", body), False))
        self.assertLess(t.index("--- slide 1 ---"), t.index("first slide"))
        self.assertLess(t.index("--- slide 2 ---"), t.index("second"))


class Epub(unittest.TestCase):
    def test_spine_order_and_html_stripped(self):
        ch1 = "<html><head><title>T</title><style>p{color:red}</style></head><body><h1>Chapter&nbsp;1</h1><p>บทที่หนึ่ง</p><script>x()</script></body></html>"
        ch2 = "<html><body><p>Second &amp; last</p></body></html>"
        t = text(documents.extract("b.epub", None, epub([("text/ch%202.xhtml", ch2), ("text/ch1.xhtml", ch1)], ["text-ch1", "text-ch2"]), False))
        self.assertIn("Chapter\u00a01", t)
        self.assertIn("บทที่หนึ่ง", t)
        self.assertNotIn("color:red", t)
        self.assertNotIn("x()", t)
        self.assertLess(t.index("บทที่หนึ่ง"), t.index("Second & last"))


class Rtf(unittest.TestCase):
    def test_control_words_go_and_the_text_stays(self):
        rtf = (rb"{\rtf1\ansi\ansicpg874\deff0{\fonttbl{\f0 Tahoma;}}{\colortbl;\red0\green0\blue0;}"
               rb"{\*\generator Riched20;}{\info{\title Secret Title}}\f0\fs20 Hello \b bold\b0  world\par "
               rb"\'ca\'c7\'d1\'ca\'b4\'d5\par \u3626?\u3623?\tab x\line y\par "
               rb"{\field{\*\fldinst HYPERLINK x}{\fldrslt link text}}\par \{braces\} and \\ backslash}")
        t = text(documents.extract("n.rtf", None, rtf, False))
        self.assertIn("Hello bold world", t)
        self.assertIn("สวัสดี", t)
        self.assertIn("สว\tx\ny", t)
        self.assertIn("link text", t)
        self.assertIn("{braces} and \\ backslash", t)
        for gone in ("Tahoma", "Secret Title", "Riched20", "HYPERLINK", "red0"):
            self.assertNotIn(gone, t)


class NotesNotExceptions(unittest.TestCase):
    OLE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\0" * 504

    def test_a_legacy_binary_office_file_says_to_save_it_again(self):
        for name, new in (("old.doc", ".docx"), ("old.xls", ".xlsx"), ("old.ppt", ".pptx")):
            t = text(documents.extract(name, None, self.OLE, False))
            self.assertIn("old binary", t)
            self.assertIn(new, t)

    def test_an_ole_file_with_a_new_name_is_called_protected_or_old(self):
        t = text(documents.extract("locked.docx", None, self.OLE, False))
        self.assertIn("password", t)

    def test_broken_and_unknown_files_give_a_note(self):
        for name, data in (("bad.docx", b"PK\x03\x04 broken"), ("bad.xlsx", zipped({"x": "y"})),
                           ("bad.epub", zipped({"mimetype": "application/epub+zip"})), ("bad.rtf", b"{\\rtf1"),
                           ("bad.docx", zipped({"word/document.xml": "<not xml"})), ("blob.bin", b"\x00\x01\x02")):
            parts = documents.extract(name, None, data, False)
            self.assertTrue(parts and all(p["type"] == "text" for p in parts), name)
            self.assertTrue(text(parts).startswith(f"[Document: {name}]"), name)

    def test_a_utf8_text_file_is_its_text(self):
        self.assertIn("plain ไทย", text(documents.extract("a.txt", "text/plain", "plain ไทย".encode(), False)))

    def test_too_large_a_file_is_refused_before_it_is_read(self):
        old = documents.MAX_BYTES
        documents.MAX_BYTES = 100
        self.addCleanup(setattr, documents, "MAX_BYTES", old)
        self.assertIn("more than", text(documents.extract("r.docx", None, docx(para("x" * 200)), False)))

    def test_a_zip_bomb_stops_at_the_byte_budget(self):
        old = documents.MAX_UNZIPPED
        documents.MAX_UNZIPPED = 10_000
        self.addCleanup(setattr, documents, "MAX_UNZIPPED", old)
        t = text(documents.extract("bomb.docx", None, docx(para("a" * 200_000)), False))
        self.assertIn("too large", t)
        self.assertNotIn("a" * 1000, t)

    def test_no_entity_from_outside_is_read(self):
        evil = ('<?xml version="1.0"?><!DOCTYPE d [<!ENTITY x SYSTEM "file:///C:/Windows/win.ini">]>'
                f'<w:document {W}><w:body><w:p><w:r><w:t>&x;</w:t></w:r></w:p></w:body></w:document>')
        t = text(documents.extract("e.docx", None, zipped({"word/document.xml": evil}), False))
        self.assertNotIn("[fonts]", t.lower())

    def test_the_text_is_cut_at_the_limit_with_a_note(self):
        old = documents.MAX_TEXT
        documents.MAX_TEXT = 50
        self.addCleanup(setattr, documents, "MAX_TEXT", old)
        t = text(documents.extract("long.docx", None, docx(para("b" * 500)), False))
        self.assertIn("[cut here", t)
        self.assertLess(t.count("b"), 60)


class Detection(unittest.TestCase):
    def test_by_name_by_content_and_by_media_type(self):
        self.assertEqual(documents.kind_of("A.DOCX", None, b""), "docx")
        self.assertEqual(documents.kind_of("", None, docx(para("x"))), "docx")
        self.assertEqual(documents.kind_of("", None, xlsx([("S", "")], [])), "xlsx")
        self.assertEqual(documents.kind_of("", None, pptx({1: ["x"]})), "pptx")
        self.assertEqual(documents.kind_of("", None, odf("text", "")), "odt")
        self.assertEqual(documents.kind_of("", None, epub([], [])), "epub")
        self.assertEqual(documents.kind_of("", None, b"{\\rtf1 x}"), "rtf")
        self.assertEqual(documents.kind_of("", None, b"%PDF-1.4"), "pdf")
        self.assertEqual(documents.kind_of("file", "application/vnd.oasis.opendocument.presentation", b"??"), "odp")
        self.assertEqual(documents.kind_of("scan", "application/pdf", b"??"), "pdf")
        self.assertIsNone(documents.kind_of("x.bin", "application/octet-stream", b"\x00"))

    @unittest.skipIf(pypdf is None, "needs pypdf")
    def test_a_pdf_with_no_media_type_is_read_by_its_name(self):
        t = text(documents.extract("r.pdf", "application/octet-stream", text_pdf(["page text"]), False))
        self.assertIn("--- page 1 ---\npage text", t)


class ForRead(unittest.TestCase):
    def test_text_of_gives_the_text_or_a_note(self):
        body, ok = documents.text_of("r.docx", docx(para("hello")))
        self.assertTrue(ok)
        self.assertEqual(body.strip(), "hello")
        note, ok = documents.text_of("old.doc", NotesNotExceptions.OLE)
        self.assertFalse(ok)
        self.assertIn(".docx", note)


class OldNamesNewContent(unittest.TestCase):
    def test_a_doc_that_is_rtf_or_html_or_a_zip_is_read(self):
        self.assertIn("rtf words", text(documents.extract("w.doc", None, rb"{\rtf1\ansi rtf words}", False)))
        self.assertIn("<td>html cell</td>", text(documents.extract("x.xls", None, b"<table><tr><td>html cell</td></tr></table>", False)))
        self.assertIn("zip body", text(documents.extract("z.doc", None, docx(para("zip body")), False)))


class InTheReadTool(unittest.TestCase):
    """Read gives a document's text by lines, as it gives a text file; an old binary Office file gives the note."""

    def setUp(self):
        import tempfile
        from serve import agent, permissions
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.proj = Path(self._tmp.name).resolve()
        self.srv = agent.AgentServer()
        self.ctx = agent.AgentContext(policy=permissions.Policy(cwd=str(self.proj)), session="s")
        self.ctx.vision = False

    def read(self, **args):
        r = self.srv.call("Read", args, ctx=self.ctx)
        return r, "\n".join(b["text"] for b in r["content"] if b.get("type") == "text")

    def test_each_format_is_read_as_numbered_lines(self):
        samples = {"r.docx": (docx(para("word body")), "word body"),
                   "b.xlsx": (xlsx([("S", '<row r="1"><c r="A1" t="s"><v>0</v></c></row>')], ["cell text"]), "cell text"),
                   "d.pptx": (pptx({1: ["slide text"]}), "slide text"),
                   "n.odt": (odf("text", "<office:text><text:p>odt text</text:p></office:text>"), "odt text"),
                   "e.epub": (epub([("c.xhtml", "<html><body><p>epub text</p></body></html>")], ["c"]), "epub text"),
                   "n.rtf": (rb"{\rtf1\ansi rtf text}", "rtf text")}
        for name, (data, want) in samples.items():
            with self.subTest(name=name):
                (self.proj / name).write_bytes(data)
                r, t = self.read(file_path=name)
                self.assertFalse(r.get("isError"), t)
                self.assertIn(want, t)
                self.assertRegex(t, r"\s*1\t")                  # numbered as a text file is

    def test_offset_and_limit_go_on_where_a_long_document_stops(self):
        (self.proj / "long.docx").write_bytes(docx("".join(para(f"line {i}") for i in range(1, 31))))
        _, t = self.read(file_path="long.docx", offset=10, limit=3)
        self.assertIn("line 10", t)
        self.assertIn("line 12", t)
        self.assertNotIn("line 13", t)
        self.assertNotIn("line 9\n", t)

    def test_an_old_binary_file_gives_the_note(self):
        (self.proj / "old.xls").write_bytes(NotesNotExceptions.OLE)
        r, t = self.read(file_path="old.xls")
        self.assertTrue(r["isError"])
        self.assertIn(".xlsx", t)

    def test_a_document_read_is_not_a_read_for_edit(self):
        (self.proj / "r.docx").write_bytes(docx(para("x")))
        self.read(file_path="r.docx")
        r = self.srv.call("Edit", {"file_path": "r.docx", "old_string": "x", "new_string": "y"}, ctx=self.ctx)
        self.assertTrue(r.get("isError"))

    def test_the_tool_names_the_formats(self):
        desc = {t["name"]: t for t in self.srv.tools}["Read"]["description"]
        for word in ("Word", "Excel", "PowerPoint", "EPUB", "RTF"):
            self.assertIn(word, desc)


if __name__ == "__main__":
    unittest.main()
