"""Reading an image or a PDF with the Read tool (serve/media.py, issue #99): the image goes to the model as an image when the server has vision; a PDF is read by pages as text, a scan as an image
(vision on) or a note (vision off); limits keep one file from taking the whole context."""
from __future__ import annotations

import base64
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from serve import agent, media, permissions  # noqa: E402
from serve.mcp import McpHub  # noqa: E402
from serve.test_agent_chat import DONE, Fixture  # noqa: E402
from serve.test_harness import tool_call  # noqa: E402

try:
    from PIL import Image
except ImportError:                                  # pragma: no cover
    Image = None

try:
    import pypdf
except ImportError:                                  # pragma: no cover
    pypdf = None


def png(size=(40, 30), colour=(200, 30, 30), fmt="PNG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, format=fmt)
    return buf.getvalue()


def text_pdf(pages: list[str | None]) -> bytes:
    """A small PDF: a page with text is written with Helvetica, one with None is blank (a scan has no text layer)."""
    objs: list[bytes] = []
    kids = []
    n = 3
    for body in pages:
        page_id = n
        if body is None:
            objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >>".encode())
            n += 1
        else:
            stream = f"BT /F1 12 Tf 20 100 Td ({body}) Tj ET".encode()
            objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents {n + 1} 0 R /Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> >>".encode())
            objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
            n += 2
        kids.append(f"{page_id} 0 R")
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    all_objs = [b"<< /Type /Catalog /Pages 2 0 R >>", f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>".encode(), *objs]
    for i, o in enumerate(all_objs, 1):
        offsets.append(out.tell())
        out.write(f"{i} 0 obj\n".encode() + o + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(all_objs) + 1}\n0000000000 65535 f \n".encode())
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(f"trailer\n<< /Size {len(all_objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return out.getvalue()


def blocks(r, kind):
    return [b for b in r["content"] if b.get("type") == kind]


def words(r) -> str:
    return "\n".join(b["text"] for b in blocks(r, "text"))


class Pages(unittest.TestCase):
    def test_a_page_a_range_or_a_list_of_them(self):
        self.assertEqual(media.parse_pages("3", 10), ([3], None))
        self.assertEqual(media.parse_pages("1-5", 10), ([1, 2, 3, 4, 5], None))
        self.assertEqual(media.parse_pages(" 2 , 4-6 ", 10), ([2, 4, 5, 6], None))
        self.assertEqual(media.parse_pages("1,1-2", 10), ([1, 2], None))                # each page once

    def test_what_is_not_a_page_range_is_refused_with_a_reason(self):
        for bad in ("", "x", "0", "0-3", "5-2", "1-", "-3", "1;2", "1-2-3", None, 3):
            pages, why = media.parse_pages(bad, 10)
            self.assertIsNone(pages, bad)
            self.assertTrue(why, bad)
        self.assertIn("goes past the end", media.parse_pages("8-12", 10)[1])
        self.assertIn("At most 20", media.parse_pages("1-21", 30)[1])

    def test_the_kind_of_a_file_is_its_extension(self):
        self.assertEqual([media.kind_of(p) for p in ("a.PNG", "b.jpeg", "c.webp", "d.pdf", "e.txt", "f")], ["image", "image", "image", "pdf", None, None])


@unittest.skipIf(Image is None, "needs Pillow")
class Images(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def file(self, name, data):
        p = self.dir / name
        p.write_bytes(data)
        return str(p)

    def test_with_vision_the_image_comes_back_as_an_image_with_a_line_about_it(self):
        p = self.file("shot.png", png())
        r = media.read_image(p, "shot.png", True)
        self.assertFalse(r.get("isError"))
        (img,) = blocks(r, "image")
        self.assertEqual(img["mimeType"], "image/png")
        self.assertEqual(base64.b64decode(img["data"])[:4], b"\x89PNG")
        self.assertIn("shot.png: 40 x 30 px", words(r))

    def test_without_vision_it_says_so_and_does_not_fail_silently(self):
        r = media.read_image(self.file("shot.png", png()), "shot.png", False)
        self.assertTrue(r["isError"])
        self.assertIn("without the vision encoder", words(r))
        self.assertEqual(blocks(r, "image"), [])

    def test_a_large_image_is_made_smaller_a_small_one_is_left_alone(self):
        big = media.read_image(self.file("big.png", png((4000, 3000))), "big.png", True)
        out = Image.open(io.BytesIO(base64.b64decode(blocks(big, "image")[0]["data"])))
        self.assertLessEqual(max(out.size), media.MAX_SIDE)
        self.assertIn("made smaller", words(big))
        small = png((100, 50))
        got = media.read_image(self.file("s.png", small), "s.png", True)
        self.assertEqual(base64.b64decode(blocks(got, "image")[0]["data"]), small)

    def test_a_webp_and_a_jpeg_are_read(self):
        w = media.read_image(self.file("a.webp", png(fmt="WEBP")), "a.webp", True)
        self.assertEqual(blocks(w, "image")[0]["mimeType"], "image/png")               # the encoder reads PNG: a webp is converted
        j = media.read_image(self.file("a.jpg", png(fmt="JPEG")), "a.jpg", True)
        self.assertEqual(blocks(j, "image")[0]["mimeType"], "image/jpeg")

    def test_a_file_that_is_not_the_image_it_says_is_refused_before_it_can_break_the_request(self):
        for name, data in (("fake.png", b"this is text"), ("fake.jpg", b"\x89PNG\r\n\x1a\nnope"), ("fake.webp", b"RIFFxxxxWEBPjunk"), ("empty.png", b"")):
            r = media.read_image(self.file(name, data), name, True)
            self.assertTrue(r["isError"], name)
            self.assertEqual(blocks(r, "image"), [], name)

    def test_too_large_a_file_is_refused(self):
        p = self.file("huge.png", b"\x89PNG\r\n\x1a\n" + b"0" * (media.MAX_IMAGE_BYTES + 1))
        r = media.read_image(p, "huge.png", True)
        self.assertTrue(r["isError"])
        self.assertIn("more than", words(r))


@unittest.skipIf(pypdf is None, "needs pypdf")
class Pdfs(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def pdf(self, pages, name="doc.pdf"):
        p = self.dir / name
        p.write_bytes(text_pdf(pages))
        return str(p)

    def test_the_text_of_each_page_in_order_under_its_number(self):
        r = media.read_pdf(self.pdf(["first page", "second page", "third page"]), "doc.pdf", None, False)
        self.assertFalse(r.get("isError"))
        t = words(r)
        self.assertIn("3 pages", t)
        self.assertLess(t.index("--- page 1 ---"), t.index("first page"))
        self.assertLess(t.index("first page"), t.index("--- page 2 ---"))
        self.assertLess(t.index("second page"), t.index("third page"))

    def test_a_range_reads_only_those_pages(self):
        r = media.read_pdf(self.pdf([f"text of page {i}" for i in range(1, 16)]), "doc.pdf", "3-4", False)
        t = words(r)
        self.assertIn("text of page 3", t)
        self.assertIn("text of page 4", t)
        self.assertNotIn("text of page 2", t)
        self.assertNotIn("text of page 5", t)

    def test_a_long_pdf_needs_pages_and_says_how(self):
        r = media.read_pdf(self.pdf([f"p{i}" for i in range(11)]), "doc.pdf", None, False)
        self.assertTrue(r["isError"])
        self.assertIn("11 pages", words(r))
        self.assertIn("pages", words(r))
        self.assertFalse(media.read_pdf(self.pdf([f"p{i}" for i in range(10)], "ten.pdf"), "ten.pdf", None, False).get("isError"))

    def test_bad_pages_and_too_many_are_refused(self):
        p = self.pdf([f"p{i}" for i in range(30)])
        self.assertIn("past the end", words(media.read_pdf(p, "doc.pdf", "28-35", False)))
        self.assertIn("At most 20", words(media.read_pdf(p, "doc.pdf", "1-21", False)))
        self.assertTrue(media.read_pdf(p, "doc.pdf", "one", False)["isError"])
        self.assertFalse(media.read_pdf(p, "doc.pdf", "1-20", False).get("isError"))

    def test_the_text_is_cut_at_a_limit_and_says_where_to_go_on(self):
        old = media.MAX_PDF_TEXT
        media.MAX_PDF_TEXT = 30
        self.addCleanup(setattr, media, "MAX_PDF_TEXT", old)
        t = words(media.read_pdf(self.pdf(["a" * 50, "second"]), "doc.pdf", None, False))
        self.assertIn("is cut here", t)
        self.assertNotIn("second", t)

    def test_a_scan_is_an_image_with_vision_in_the_place_of_its_page(self):
        r = media.read_pdf(self.pdf(["text page", None, "after the scan"]), "doc.pdf", None, True)
        kinds = [b["type"] for b in r["content"]]
        self.assertEqual(kinds, ["text", "image", "text"])
        self.assertIn("--- page 2 (a scan with no text layer, shown as an image) ---", r["content"][0]["text"])
        self.assertIn("after the scan", r["content"][2]["text"])
        self.assertEqual(base64.b64decode(r["content"][1]["data"])[:4], b"\x89PNG")

    def test_a_scan_without_vision_is_a_note_and_not_a_silence(self):
        r = media.read_pdf(self.pdf(["text page", None]), "doc.pdf", None, False)
        self.assertEqual(blocks(r, "image"), [])
        self.assertIn("page 2 --- (no text layer; this server has no vision encoder", words(r))

    def test_only_a_few_scans_are_given_as_images_at_once(self):
        r = media.read_pdf(self.pdf([None] * 10), "doc.pdf", None, True)
        self.assertEqual(len(blocks(r, "image")), media.MAX_IMAGE_PAGES)
        self.assertIn(f"only {media.MAX_IMAGE_PAGES} scanned pages", words(r))

    def test_a_password_protected_pdf_and_a_broken_one_say_so(self):
        w = pypdf.PdfWriter()
        w.add_blank_page(100, 100)
        w.encrypt("secret")
        buf = io.BytesIO()
        w.write(buf)
        p = self.dir / "locked.pdf"
        p.write_bytes(buf.getvalue())
        self.assertIn("password", words(media.read_pdf(str(p), "locked.pdf", None, False)))
        q = self.dir / "broken.pdf"
        q.write_bytes(b"%PDF-1.4 this is not a pdf")
        r = media.read_pdf(str(q), "broken.pdf", None, False)
        self.assertTrue(r["isError"])
        self.assertIn("could not be read", words(r))


@unittest.skipIf(Image is None, "needs Pillow")
class InTheTool(unittest.TestCase):
    """Read gives an image or a PDF by what the chat can do (ctx.vision), and the rules are the same as for any file."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.proj = Path(self._tmp.name).resolve()
        (self.proj / "shot.png").write_bytes(png())
        self.srv = agent.AgentServer()

    def ctx(self, vision):
        c = agent.AgentContext(policy=permissions.Policy(cwd=str(self.proj)), session="s")
        c.vision = vision
        return c

    def test_read_gives_the_image_when_there_is_vision(self):
        r = self.srv.call("Read", {"file_path": "shot.png"}, ctx=self.ctx(True))
        self.assertFalse(r.get("isError"))
        self.assertEqual(len(blocks(r, "image")), 1)

    def test_read_says_why_when_there_is_none(self):
        r = self.srv.call("Read", {"file_path": "shot.png"}, ctx=self.ctx(False))
        self.assertTrue(r["isError"])
        self.assertIn("vision", words(r))

    def test_the_tool_offers_pages_and_says_what_it_reads(self):
        read = {t["name"]: t for t in self.srv.tools}["Read"]
        self.assertIn("pages", read["inputSchema"]["properties"])
        self.assertIn("image", read["description"])
        self.assertIn("PDF", read["description"])

    def test_the_hub_hands_the_image_over_with_the_result_and_not_as_text(self):
        hub = McpHub({}, builtins={"agent": self.srv})
        r = hub.call("Read", {"file_path": "shot.png"}, ctx=self.ctx(True))
        self.assertTrue(r["ok"])
        self.assertEqual(len(r["images"]), 1)
        self.assertTrue(r["images"][0].startswith("data:image/png;base64,"))
        self.assertNotIn("base64", r["text"])
        self.assertIn("shot.png", r["text"])


@unittest.skipIf(Image is None, "needs Pillow")
class InTheChat(Fixture):
    """The model reads an image through the chat endpoint: the vision encoder is given it as an image in the next turn."""

    class FakeVision:
        def __init__(self, d):
            self.dir = Path(d)
            self.rows = self.dir / "img.sve"
            self.rows.write_bytes(b"rows")
            self.seen: list[str] = []

        def encode(self, source):
            self.seen.append(source)
            return self.rows, 3

    def look(self, vision):
        (self.proj / "shot.png").write_bytes(png())
        self.start(tool_call("Read", file_path="shot.png"), DONE)
        if vision:
            self.fake = self.FakeVision(self.base)
            self.svc.vision = self.fake
        return self.chat(self.body(messages=[{"role": "user", "content": "what is in shot.png?"}]))

    def test_the_image_reaches_the_vision_encoder_in_the_next_turn_and_the_page_gets_only_a_line(self):
        status, said, events = self.look(True)
        self.assertEqual(status, 200)
        self.assertTrue(said.strip().endswith("Done."))
        (res,) = [e for e in events if e["event"] == "result"]
        self.assertTrue(res["ok"])
        self.assertIn("shot.png: 40 x 30 px", res["text"])
        self.assertNotIn("base64", json.dumps(events))                           # the page is not sent the picture again
        self.assertEqual(len(self.fake.seen), 1)
        self.assertTrue(self.fake.seen[0].startswith("data:image/png;base64,"))

    def test_without_vision_the_model_is_told_and_nothing_breaks(self):
        status, said, events = self.look(False)
        self.assertEqual(status, 200)
        (res,) = [e for e in events if e["event"] == "result"]
        self.assertFalse(res["ok"])
        self.assertIn("vision", res["text"])
        self.assertTrue(said.strip().endswith("Done."))


if __name__ == "__main__":
    unittest.main()
