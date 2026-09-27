"""Anthropic document blocks must survive request conversion in page order."""

import base64
from pathlib import Path
import unittest

from serve.frontend import anthropic_to_messages, images_of


PDF = Path(__file__).with_name("fixtures") / "sample.pdf"


def request(source):
    return {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "Read this: "},
        {"type": "document", "title": "sample.pdf", "source": source},
        {"type": "text", "text": " Summarize it."},
    ]}]}


class DocumentBlocksTest(unittest.TestCase):
    def test_pdf_text_and_scanned_page_note_preserve_order(self):
        source = {"type": "base64", "media_type": "application/pdf",
                  "data": base64.b64encode(PDF.read_bytes()).decode()}
        messages, _, _ = anthropic_to_messages(request(source))
        content = messages[0]["content"]
        self.assertIn("Read this:", content)
        self.assertIn("Revenue is 42.", content)
        self.assertIn("page 2", content.lower())
        self.assertLess(content.index("Read this:"), content.index("Revenue is 42."))
        self.assertLess(content.index("Revenue is 42."), content.index("Summarize it."))

    def test_scanned_page_becomes_image_with_vision(self):
        source = {"type": "base64", "media_type": "application/pdf",
                  "data": base64.b64encode(PDF.read_bytes()).decode()}
        messages, _, _ = anthropic_to_messages(request(source), vision=True)
        self.assertEqual(len(images_of(messages)), 1)
        self.assertTrue(images_of(messages)[0].startswith("data:image/png;base64,"))
        self.assertIn("Revenue is 42.", str(messages[0]["content"]))

    def test_bad_pdf_is_visible(self):
        source = {"type": "base64", "media_type": "application/pdf", "data": "%%%"}
        messages, _, _ = anthropic_to_messages(request(source))
        self.assertIn("could not", str(messages[0]["content"]).lower())

    def test_plain_text_document(self):
        messages, _, _ = anthropic_to_messages(request({"type": "text", "media_type": "text/plain",
                                                          "data": "Plain document body"}))
        self.assertIn("Plain document body", messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
