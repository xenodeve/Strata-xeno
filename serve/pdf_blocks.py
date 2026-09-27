"""Anthropic `document` blocks -> what the local model can take (2026-09-26). Ours.

Claude Code's Read sends a PDF as a `document` block; the translation dropped it as
an unknown block, so the model saw nothing. Here: the text layer of every page, in
order, under the document's title; a page with no text layer (a scan) becomes an
image for the vision tower when one is loaded, else a one-line note. A document that
cannot be parsed says so -- it never vanishes silently.
"""
import base64
import io

MAX_IMAGE_PAGES = 8          # scanned pages sent as images, at most
RENDER_SCALE = 1.5           # 72 dpi * 1.5 = 108 dpi: readable text, modest tokens


def document_text(block):
    """Plain-text documents (source type text / content)."""
    src = block.get("source") or {}
    if src.get("type") == "text":
        return src.get("data") or ""
    if src.get("type") == "content":
        return "\n".join(c.get("text", "") for c in src.get("content") or [] if isinstance(c, dict))
    return ""


def _header(block, pages = None):
    title = block.get("title") or "document"
    return f"[Document: {title}" + (f", {pages} page(s)" if pages else "") + "]"


def document_parts(block, vision = False):
    """A list of OpenAI content parts (text and, with vision, image_url) for one block."""
    src = block.get("source") or {}
    if src.get("type") in ("text", "content"):
        return [{"type": "text", "text": _header(block) + "\n" + document_text(block)}]
    if src.get("type") != "base64" or "pdf" not in (src.get("media_type") or ""):
        return [{"type": "text", "text": _header(block) + f"\n(unsupported document source: {src.get('type')})"}]
    try:
        data = base64.b64decode(src.get("data") or "")
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        texts = [(p.extract_text() or "").strip() for p in reader.pages]
    except Exception as e:
        return [{"type": "text", "text": _header(block) + f"\n(the PDF could not be read: {type(e).__name__})"}]
    parts = [{"type": "text", "text": _header(block, len(texts))}]
    images = 0
    for i, t in enumerate(texts, 1):
        if t:
            parts.append({"type": "text", "text": f"--- page {i} ---\n{t}"})
            continue
        if vision and images < MAX_IMAGE_PAGES:
            url = _page_image(data, i - 1)
            if url:
                images += 1
                parts.append({"type": "text", "text": f"--- page {i} (scanned, as an image) ---"})
                parts.append({"type": "image_url", "image_url": {"url": url}})
                continue
        parts.append({"type": "text", "text": f"--- page {i} --- (no text layer"
                      + ("" if vision else "; images are not enabled on this server") + ")"})
    return parts


def _page_image(data, index):
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(data)
        img = pdf[index].render(scale = RENDER_SCALE).to_pil()
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format = "PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None
