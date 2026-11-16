"""desktop/test_words.py - the app's one table of words, English and Thai (#310 part 3 task 2).

    python -m pytest desktop/test_words.py -q
"""
from __future__ import annotations

import string

import pytest

from desktop import words


def _fields(text: str) -> set[str]:
    return {f for _, f, _, _ in string.Formatter().parse(text) if f is not None}


def test_every_key_has_thai():
    for key, row in words.TABLE.items():
        assert set(row) == {"en", "th"}, key
        assert row["en"].strip() and row["th"].strip(), key
        assert row["th"] != row["en"] or not any(c.isalpha() for c in row["en"]), key


def test_format_keys_match():
    for key, row in words.TABLE.items():
        assert _fields(row["en"]) == _fields(row["th"]), key


def test_exact_copy():
    assert words.TABLE["plain_http"]["en"] == (
        "This address is plain http to another network: your token and the files your PC sends travel unencrypted. "
        "Use Tailscale, Radmin VPN or an https tunnel.")
    assert words.TABLE["pending"]["en"] == "Waiting for approval on {server}: code {code}"
    assert words.TABLE["pending"]["th"] == "รออนุมัติที่ {server}: รหัส {code}"
    assert words.TABLE["plain_http"]["th"] == (
        "ที่อยู่นี้เป็น http ธรรมดาไปยังเครือข่ายอื่น โทเค็นของคุณและไฟล์ที่เครื่องคุณส่งไปจะวิ่งแบบไม่เข้ารหัส "
        "ใช้ Tailscale, Radmin VPN หรือ tunnel แบบ https แทน")
    assert words.TABLE["update"]["th"] == "อัปเดต Strata Desktop: {server} ใช้โปรโตคอล runner รุ่น {n}"
    assert words.TABLE["update"]["en"] == "Update Strata Desktop: {server} speaks runner protocol {n}."
    assert words.TABLE["no_webview2"]["en"] == (
        "Strata Desktop needs the Microsoft Edge WebView2 Runtime, which this PC does not have. Install it from "
        "https://go.microsoft.com/fwlink/p/?LinkId=2124703 and start Strata Desktop again.")
    assert "https://go.microsoft.com/fwlink/p/?LinkId=2124703" in words.TABLE["no_webview2"]["th"]


def test_t_formats_in_the_current_language(monkeypatch):
    monkeypatch.setattr(words, "LANG", "en")
    assert words.t("pending", server="lab", code="K7QZ") == "Waiting for approval on lab: code K7QZ"
    monkeypatch.setattr(words, "LANG", "th")
    assert words.t("pending", server="lab", code="K7QZ") == "รออนุมัติที่ lab: รหัส K7QZ"


def test_unknown_key_raises():
    with pytest.raises(KeyError):
        words.t("no_such_words")


def test_lang_is_en_or_th():
    assert words.LANG in ("en", "th")
    assert words.pick_lang(0x041E) == "th"            # Thai (Thailand)
    assert words.pick_lang(0x0409) == "en"            # English (United States)
    assert words.pick_lang(0x0809) == "en"
    assert words.pick_lang(None) == "en"
