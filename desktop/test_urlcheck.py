"""desktop/test_urlcheck.py - the plain-http warning (#310 part 3 task 2): quiet for https and for http that stays on
this PC or inside a VPN (Radmin 26/8, Tailscale 100.64/10 and *.ts.net); a warning for every other http address.

    python -m pytest desktop/test_urlcheck.py -q
"""
from __future__ import annotations

import pytest

from desktop import urlcheck, words


def test_https_never_warns():
    for url in ("https://example.com", "https://192.168.1.10:8095", "https://25.255.255.255", "HTTPS://x.y/p"):
        assert urlcheck.plain_http_warning(url) is None, url


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:8095", "http://localhost:8095", "http://[::1]:8095", "http://26.33.142.160:8095",
    "http://100.64.0.1", "http://100.127.255.254", "http://strata.tail1234.ts.net",
    # beyond the brief: case, a path, 127/8 beyond .0.0.1, the network edges
    "HTTP://LOCALHOST:8095/", "http://127.255.255.254/strata", "http://26.0.0.0", "http://26.255.255.255",
    "http://Strata.Tail1234.TS.NET:8095", "  http://127.0.0.1:8095  ", "http://[::ffff:127.0.0.1]:8095",
])
def test_quiet_cases(url):
    assert urlcheck.plain_http_warning(url) is None


@pytest.mark.parametrize("url", [
    "http://192.168.1.10:8095", "http://100.128.0.1", "http://25.255.255.255", "http://example.com",
    # beyond the brief: names that only look local, and other private ranges
    "http://127.0.0.1@example.com", "http://127.0.0.1.example.com", "http://evilts.net", "http://ts.net.example.com",
    "http://localhost.example.com", "http://10.0.0.5:8095", "http://27.0.0.1", "http://100.63.255.255",
    "http://[fe80::1]:8095", "http://[::ffff:192.168.1.10]",
])
def test_warn_cases(url):
    assert urlcheck.plain_http_warning(url) == words.t("plain_http")


@pytest.mark.parametrize("url", ["", "   ", "http://", "http://[::1", "ftp://example.com", "nonsense", None, 5])
def test_unfinished_text_is_quiet(url):
    # text as it is typed: an unfinished address gets no warning yet (the save refuses it on its own terms)
    assert urlcheck.plain_http_warning(url) is None


def test_bad_port_does_not_raise():
    # the port is never read: a port out of range is the save's to refuse; the host alone decides the warning
    assert urlcheck.plain_http_warning("http://example.com:99999") == words.t("plain_http")
    assert urlcheck.plain_http_warning("http://127.0.0.1:99999") is None


def test_follows_the_language(monkeypatch):
    monkeypatch.setattr(words, "LANG", "th")
    assert urlcheck.plain_http_warning("http://example.com") == words.TABLE["plain_http"]["th"]
    monkeypatch.setattr(words, "LANG", "en")
    assert urlcheck.plain_http_warning("http://example.com") == words.TABLE["plain_http"]["en"]
