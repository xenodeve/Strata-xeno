"""serve/media.py - what the Read tool does with an image or a PDF (issue #99), as Claude Code's Read does.

An image is given to the model as an image in its next turn when the server has the vision encoder (it is made smaller first when it is large); without vision the tool says so, and
never fails silently.  A PDF is read by page range as text (a long one needs `pages`, at most 20 pages at once); a page that has no text layer - a scan - is given as an image when vision is
on, a few at most, else a note says so.  Size and page limits keep one file from taking the whole context.  The result is an MCP tool result: text blocks and `image` blocks
(base64 data and a MIME type) that serve/mcp.py hands to the model's turn.
"""
from __future__ import annotations

import base64
import io
import os
import re

IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".bmp": "image/bmp", ".webp": "image/webp"}
MAGIC = {"image/png": (b"\x89PNG\r\n\x1a\n",), "image/jpeg": (b"\xff\xd8\xff",), "image/gif": (b"GIF87a", b"GIF89a"), "image/bmp": (b"BM",), "image/webp": ()}
NEEDS_PILLOW = {"image/webp"}                    # the vision encoder reads PNG, JPEG, BMP and GIF itself
MAX_IMAGE_BYTES = 5_000_000
MAX_SIDE = 1568                                  # a larger image is made smaller: more would only cost context
MAX_PDF_BYTES = 40_000_000
WHOLE_PDF_PAGES = 10                             # a longer PDF has to be read by `pages`
MAX_PAGES = 20                                   # pages in one read
MAX_IMAGE_PAGES = 8                              # scanned pages given as images in one read
MAX_PDF_TEXT = 60_000                            # characters of text in one read


def kind_of(path: str) -> str | None:
    ext = os.path.splitext(path)[1].lower()
    return "image" if ext in IMAGE_TYPES else "pdf" if ext == ".pdf" else None


def _text(s: str) -> dict:
    return {"type": "text", "text": s}


def _err(s: str) -> dict:
    return {"content": [_text(s)], "isError": True}


def _image_block(data: bytes, mime: str) -> dict:
    return {"type": "image", "data": base64.b64encode(data).decode("ascii"), "mimeType": mime}


# ------------------------------------------------------------------------------------------------ images
def _shrunk(data: bytes, mime: str) -> tuple[bytes, str, tuple[int, int] | None, bool]:
    """(the image to give, its type, its size as (width, height) when it could be read, whether it was made smaller).  Without Pillow it is given as it is."""
    try:
        from PIL import Image
    except ImportError:
        return data, mime, None, False
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception:  # noqa: BLE001 - not an image Pillow reads: the encoder will say so
        return data, mime, None, False
    size = im.size
    if max(size) <= MAX_SIDE and mime != "image/webp":
        return data, mime, size, False
    if max(size) > MAX_SIDE:
        im.thumbnail((MAX_SIDE, MAX_SIDE))
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        im = bg
    elif im.mode != "RGB":
        im = im.convert("RGB")
    out = io.BytesIO()
    im.save(out, format="PNG")
    return out.getvalue(), "image/png", size, max(size) > MAX_SIDE


def read_image(path: str, name: str, vision: bool) -> dict:
    if not vision:
        return _err(f"{name} is an image, and this server was started without the vision encoder, so the model cannot look at images. "
                    "(Run setup again and choose vision; PDFs with a text layer and text files can still be read.)")
    size = os.path.getsize(path)
    if size > MAX_IMAGE_BYTES:
        return _err(f"The image is {size:,} bytes, more than the {MAX_IMAGE_BYTES:,} that can be read. Make it smaller first.")
    if size == 0:
        return _err("The image file is empty.")
    mime = IMAGE_TYPES[os.path.splitext(path)[1].lower()]
    with open(path, "rb") as f:
        data = f.read()
    data, mime, dims, smaller = _shrunk(data, mime)
    if dims is None:                                                    # Pillow could not read it, or is not here: only what the encoder itself reads, and only if it looks like it
        if mime in NEEDS_PILLOW:
            return _err(f"{name} is in a format that needs Pillow to be read (python -m pip install pillow), which is not installed here, or it is not a readable image.")
        if not any(data.startswith(m) for m in MAGIC[mime]):
            return _err(f"{name} is not a readable {mime.split('/')[1].upper()} image.")
    about = f"{dims[0]} x {dims[1]} px, " if dims else ""
    return {"content": [_text(f"Image {name}: {about}{size:,} bytes{', made smaller to fit' if smaller else ''}. It is shown to you with this result."), _image_block(data, mime)]}


# ------------------------------------------------------------------------------------------------ PDFs
def parse_pages(spec, total: int) -> tuple[list[int] | None, str | None]:
    """"3", "1-5", "2,4-6" -> the page numbers (1 is the first), or (None, why not)."""
    if not isinstance(spec, str) or not re.fullmatch(r"\s*\d+\s*(-\s*\d+\s*)?(,\s*\d+\s*(-\s*\d+\s*)?)*", spec):
        return None, 'pages is a page, or a range, or a list of them: "3", "1-5", "2,4-6"'
    pages: list[int] = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        a, b = int(lo), int(hi or lo)
        if a < 1 or b < a:
            return None, f"{part.strip()} is not a range of pages (the first page is 1)"
        if b > total:
            return None, f"The PDF has {total} pages; {part.strip()} goes past the end"
        pages += [p for p in range(a, b + 1) if p not in pages]
        if len(pages) > MAX_PAGES:
            return None, f"At most {MAX_PAGES} pages can be read at once"
    return pages, None


def read_pdf(path: str, name: str, pages_spec, vision: bool) -> dict:
    size = os.path.getsize(path)
    if size > MAX_PDF_BYTES:
        return _err(f"The PDF is {size:,} bytes, more than the {MAX_PDF_BYTES:,} that can be read.")
    try:
        from pypdf import PdfReader
    except ImportError:
        return _err("Reading a PDF needs pypdf (python -m pip install pypdf), which is not installed here.")
    try:
        with open(path, "rb") as f:
            data = f.read()
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            return _err(f"{name} is protected by a password; it cannot be read.")
        total = len(reader.pages)
    except Exception as e:  # noqa: BLE001 - a PDF that cannot be parsed says so
        return _err(f"The PDF could not be read ({type(e).__name__}).")
    if pages_spec in (None, ""):
        if total > WHOLE_PDF_PAGES:
            return _err(f"{name} has {total} pages: too many to read at once. Give pages, for example \"1-5\" (at most {MAX_PAGES} pages a time).")
        pages = list(range(1, total + 1))
    else:
        pages, why = parse_pages(pages_spec, total)
        if why:
            return _err(why)
    blocks: list[dict] = []
    text_chars, images, cut = 0, 0, False
    out = [f"PDF {name}: {total} page{'s' if total != 1 else ''}; reading {pages[0]}-{pages[-1]}" if len(pages) > 1 else f"PDF {name}: {total} page{'s' if total != 1 else ''}; reading page {pages[0]}"]
    for n in pages:
        try:
            body = (reader.pages[n - 1].extract_text() or "").strip()
        except Exception:  # noqa: BLE001 - one page that cannot be read does not lose the others
            body = ""
        if body:
            if text_chars + len(body) > MAX_PDF_TEXT:
                body = body[: max(0, MAX_PDF_TEXT - text_chars)]
                cut = True
            text_chars += len(body)
            out.append(f"--- page {n} ---\n{body}")
            if cut:
                out.append(f"[the text is cut here at {MAX_PDF_TEXT:,} characters; read the next pages with a new pages range]")
                break
            continue
        if vision and images < MAX_IMAGE_PAGES:
            from serve import pdf_blocks
            url = pdf_blocks._page_image(data, n - 1)
            if url:
                images += 1
                out.append(f"--- page {n} (a scan with no text layer, shown as an image) ---")
                blocks.append({"at": len(out), "image": _image_block(base64.b64decode(url.split(",", 1)[1]), "image/png")})
                out.append("")                                                  # the place the image takes
                continue
        out.append(f"--- page {n} --- (no text layer" + ("; " + (f"only {MAX_IMAGE_PAGES} scanned pages are shown as images at once" if vision else "this server has no vision encoder, so a scan cannot be read")) + ")")
    content: list[dict] = []
    buf: list[str] = []
    placed = {b["at"]: b["image"] for b in blocks}
    for i, line in enumerate(out):
        if i in placed:
            content += [_text("\n\n".join(buf))] if buf else []
            buf = []
            content.append(placed[i])
            continue
        buf.append(line)
    if buf:
        content.append(_text("\n\n".join(x for x in buf if x)))
    return {"content": content}
