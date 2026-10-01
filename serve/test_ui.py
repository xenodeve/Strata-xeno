"""serve/test_ui.py - the new web app (serve/ui, xeno UI S1): dist is built from the source, and the routes.

    python -m unittest serve.test_ui -v
"""
from __future__ import annotations

import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402
from serve.ui_hash import UI, source_hash  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class DistMatchesSource(unittest.TestCase):
    def test_dist_is_not_stale(self):
        stamp = UI / "dist" / "source-hash.txt"
        self.assertTrue(stamp.is_file(), "serve/ui/dist is not built: cd serve/ui && bun run build")
        self.assertEqual(stamp.read_text(encoding="utf-8").strip(), source_hash(),
                         "serve/ui/src changed since dist was built: cd serve/ui && bun run build, then commit dist/")

    def test_dist_uses_relative_urls(self):               # a path-prefixed reverse proxy (upstream f0e1b9e)
        html = (UI / "dist" / "index.html").read_text(encoding="utf-8")
        for bad in ('src="/', 'href="/'):
            self.assertNotIn(bad, html)
        css = next((UI / "dist" / "assets").glob("*.css")).read_text(encoding="utf-8")
        self.assertNotIn("url(/", css)


class Routes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.svc = Service(MockEngine(tok, "hi", max_context=4096), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def get(self, path):
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *a, **k):
                return None
        try:
            with urllib.request.build_opener(NoRedirect).open(self.base + path, timeout=10) as r:
                return r.status, r.headers, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers, e.read()

    def test_root_is_still_the_classic_app(self):
        code, _, body = self.get("/")
        self.assertEqual(code, 200)
        self.assertIn(b'"web/app.js"', body)

    def test_classic_is_the_same_app_under_a_prefix(self):
        code, _, body = self.get("/classic/")
        self.assertEqual(code, 200)
        self.assertIn(b'"web/app.js"', body)
        for path, want in (("/classic/web/app.js", "javascript"), ("/classic/web/app.css", "text/css"),
                           ("/classic/web/sprite.svg", "image/svg+xml")):
            with self.subTest(path=path):
                code, headers, _ = self.get(path)
                self.assertEqual(code, 200)
                self.assertIn(want, headers["Content-Type"])
        self.assertEqual(self.get("/classic/metrics")[0], 200)          # its relative API calls land on the API

    def test_a_missing_slash_redirects_relatively(self):
        for name in ("classic", "next"):
            code, headers, _ = self.get("/" + name)
            self.assertEqual((code, headers["Location"]), (301, name + "/"))

    def test_next_serves_the_built_app(self):
        code, headers, body = self.get("/next/")
        self.assertEqual(code, 200)
        self.assertIn("text/html", headers["Content-Type"])
        self.assertIn(b'id="root"', body)
        asset = next((UI / "dist" / "assets").glob("*.js")).name
        code, headers, _ = self.get("/next/assets/" + asset)
        self.assertEqual(code, 200)
        self.assertIn("javascript", headers["Content-Type"])
        self.assertIn("immutable", headers["Cache-Control"])            # hashed names never change
        font = next((UI / "dist" / "assets").glob("*.woff2")).name
        self.assertEqual(self.get("/next/assets/" + font)[1]["Content-Type"], "font/woff2")

    def test_next_serves_only_dist(self):
        for path in ("/next/assets/..%2F..%2Fpackage.json", "/next/assets/..%5C..%5Cpackage.json", "/next/package.json",
                     "/next/src/main.tsx", "/next/assets/missing.js", "/next/assets/x.exe", "/next/source-hash.txt"):
            with self.subTest(path=path):
                self.assertEqual(self.get(path)[0], 404)


if __name__ == "__main__":
    unittest.main()
