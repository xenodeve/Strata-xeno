"""Strata Desktop's words, English and Thai, in one table (#310 part 3).

`t(key, **fields)` formats the row of `key` in `LANG`, read at call time.  `LANG` is "th" when Windows' display
language is Thai (`GetUserDefaultUILanguage() & 0x3FF == 0x1E`, LANG_THAI), else "en".  Rows the plan gives exact
copy for (plain_http, pending, update, no_webview2, the tray's English) are word for word; every row carries the same
`{fields}` in both languages (desktop/test_words.py checks it).
"""
from __future__ import annotations

import os

LANG_THAI = 0x1E

TABLE: dict[str, dict[str, str]] = {
    # ---------------------------------------------------------------- the servers window: the address as typed
    "plain_http": {
        "en": "This address is plain http to another network: your token and the files your PC sends travel "
              "unencrypted. Use Tailscale, Radmin VPN or an https tunnel.",
        "th": "ที่อยู่นี้เป็น http ธรรมดาไปยังเครือข่ายอื่น โทเค็นของคุณและไฟล์ที่เครื่องคุณส่งไปจะวิ่งแบบไม่เข้ารหัส "
              "ใช้ Tailscale, Radmin VPN หรือ tunnel แบบ https แทน",
    },
    # ---------------------------------------------------------------- the server test (desktop/probe.py)
    "bad_url": {
        "en": "Enter an address that starts with http:// or https://.",
        "th": "ใส่ที่อยู่ที่ขึ้นต้นด้วย http:// หรือ https://",
    },
    "unreachable": {
        "en": "The server could not be reached. Check the address, and that the server and your VPN are on.",
        "th": "ติดต่อเซิร์ฟเวอร์ไม่ได้ ตรวจที่อยู่ และดูว่าเซิร์ฟเวอร์กับ VPN เปิดอยู่",
    },
    "timed_out": {
        "en": "The server did not answer in time.",
        "th": "เซิร์ฟเวอร์ไม่ตอบภายในเวลาที่รอ",
    },
    "tls_failed": {
        "en": "The https connection was refused: the server's certificate could not be checked.",
        "th": "เชื่อมต่อ https ไม่ได้ ตรวจใบรับรองของเซิร์ฟเวอร์ไม่ผ่าน",
    },
    "not_strata": {
        "en": "This address answers, but not as a Strata server.",
        "th": "ที่อยู่นี้ตอบกลับ แต่ไม่ใช่เซิร์ฟเวอร์ Strata",
    },
    "cannot_send": {
        "en": "The token or this PC's name has a character a request cannot carry. Check the token.",
        "th": "โทเค็นหรือชื่อเครื่องนี้มีตัวอักษรที่ส่งไปในคำขอไม่ได้ ตรวจโทเค็นอีกครั้ง",
    },
    "token_refused": {
        "en": "The server refused this token (wrong or revoked).",
        "th": "เซิร์ฟเวอร์ไม่รับโทเค็นนี้ (ผิด หรือถูกเพิกถอนแล้ว)",
    },
    "device_refused": {
        "en": "The owner of this server refused this PC. Ask them to clear it in Settings > Keys.",
        "th": "เจ้าของเซิร์ฟเวอร์นี้ปฏิเสธเครื่องนี้ ขอให้เขาล้างรายการนี้ที่ Settings > Keys",
    },
    "server_answered": {
        "en": "The server answered {code}.",
        "th": "เซิร์ฟเวอร์ตอบกลับ {code}",
    },
    "role_owner": {"en": "owner", "th": "เจ้าของ"},
    "role_guest": {"en": "guest", "th": "ผู้ใช้ร่วม"},
    "engine_down": {
        "en": "The server is up, but its model is not running.",
        "th": "เซิร์ฟเวอร์เปิดอยู่ แต่โมเดลไม่ได้ทำงาน",
    },
    # ---------------------------------------------------------------- the runner's states (tray, servers window)
    "pending": {
        "en": "Waiting for approval on {server}: code {code}",
        "th": "รออนุมัติที่ {server}: รหัส {code}",
    },
    "update": {
        "en": "Update Strata Desktop: {server} speaks runner protocol {n}.",
        "th": "อัปเดต Strata Desktop: {server} ใช้โปรโตคอล runner รุ่น {n}",
    },
    "tray_idle": {
        "en": "Strata Desktop - not connected",
        "th": "Strata Desktop - ยังไม่ได้เชื่อมต่อ",
    },
    "tray_connecting": {
        "en": "Connecting to {server}…",
        "th": "กำลังเชื่อมต่อ {server}…",
    },
    "tray_online": {
        "en": "Online - {server} as {who}",
        "th": "ออนไลน์ - {server} ในชื่อ {who}",
    },
    "tray_offline": {
        "en": "Offline - {server} cannot be reached; trying again",
        "th": "ออฟไลน์ - ติดต่อ {server} ไม่ได้ กำลังลองใหม่",
    },
    "tray_refused_key": {
        "en": "The key for {server} was refused (wrong or revoked)",
        "th": "{server} ไม่รับคีย์ (ผิด หรือถูกเพิกถอนแล้ว)",
    },
    "tray_refused_device": {
        "en": "The owner of {server} refused this PC",
        "th": "เจ้าของ {server} ปฏิเสธเครื่องนี้",
    },
    "tray_refused_replaced": {
        "en": "Another Strata runner on this PC took over",
        "th": "Strata runner อีกตัวบนเครื่องนี้เข้ามาทำงานแทนแล้ว",
    },
    "tray_refused": {
        "en": "{server} refused the runner",
        "th": "{server} ไม่รับ runner",
    },
    "tray_error": {
        "en": "My PC's tools could not start: {message}",
        "th": "เครื่องมือของ My PC เริ่มทำงานไม่ได้: {message}",
    },
    "tray_no_shell": {
        "en": " (no shell: Git for Windows was not found, Bash is off)",
        "th": " (ไม่มี shell: ไม่พบ Git for Windows จึงปิด Bash)",
    },
    # ---------------------------------------------------------------- the tray's menu
    "menu_open": {"en": "Open", "th": "เปิด"},
    "menu_switch": {"en": "Switch server", "th": "เปลี่ยนเซิร์ฟเวอร์"},
    "menu_copy_code": {"en": "Copy code", "th": "คัดลอกรหัส"},
    "menu_quit": {"en": "Quit", "th": "ออก"},
    # ---------------------------------------------------------------- the servers window's labels
    "label_name": {"en": "Name", "th": "ชื่อ"},
    "label_url": {"en": "Address", "th": "ที่อยู่"},
    "label_token": {"en": "Token", "th": "โทเค็น"},
    "token_set": {"en": "token set", "th": "ตั้งโทเค็นแล้ว"},
    "token_missing": {"en": "no token", "th": "ยังไม่มีโทเค็น"},
    "token_unreadable": {
        "en": "token unreadable (saved by another Windows user): enter it again",
        "th": "อ่านโทเค็นไม่ได้ (บันทึกโดยผู้ใช้ Windows คนอื่น) ใส่ใหม่อีกครั้ง",
    },
    "token_kept": {
        "en": "Your token is kept encrypted for your Windows user; while a server is open the web app holds it as a "
              "browser would.",
        "th": "โทเค็นของคุณถูกเก็บแบบเข้ารหัสไว้สำหรับผู้ใช้ Windows ของคุณ ระหว่างที่เปิดเซิร์ฟเวอร์อยู่ "
              "เว็บแอปจะถือโทเค็นไว้แบบเดียวกับเบราว์เซอร์",
    },
    "button_save": {"en": "Save", "th": "บันทึก"},
    "button_test": {"en": "Test", "th": "ทดสอบ"},
    "button_connect": {"en": "Connect", "th": "เชื่อมต่อ"},
    "button_remove": {"en": "Remove", "th": "ลบ"},
    "button_open_without_runner": {"en": "Open without My PC", "th": "เปิดโดยไม่ใช้ My PC"},
    "window_not_prepared": {
        "en": "The window could not be prepared.",
        "th": "เตรียมหน้าต่างไม่สำเร็จ",
    },
    # ---------------------------------------------------------------- notices (Task 5)
    "notice_device_title": {"en": "A device asks to be let in", "th": "มีอุปกรณ์ขอเข้าใช้"},
    "notice_device_text": {
        "en": "{key} on {label} ({addr}), code {code}. Open Strata Desktop, Settings > Keys, to decide.",
        "th": "{key} บน {label} ({addr}) รหัส {code} เปิด Strata Desktop ที่ Settings > Keys เพื่อตัดสิน",
    },
    # ---------------------------------------------------------------- start-up
    "no_webview2": {
        "en": "Strata Desktop needs the Microsoft Edge WebView2 Runtime, which this PC does not have. Install it from "
              "https://go.microsoft.com/fwlink/p/?LinkId=2124703 and start Strata Desktop again.",
        "th": "Strata Desktop ต้องใช้ Microsoft Edge WebView2 Runtime ซึ่งเครื่องนี้ยังไม่มี ติดตั้งได้จาก "
              "https://go.microsoft.com/fwlink/p/?LinkId=2124703 แล้วเปิด Strata Desktop ใหม่อีกครั้ง",
    },
}


def pick_lang(langid: int | None) -> str:
    """"th" for a Windows LANGID whose primary language is Thai, else "en"."""
    return "th" if isinstance(langid, int) and langid & 0x3FF == LANG_THAI else "en"


def _windows_ui_langid() -> int | None:
    if os.name != "nt":
        return None
    try:
        import ctypes
        f = ctypes.windll.kernel32.GetUserDefaultUILanguage
        f.restype = ctypes.c_ushort
        f.argtypes = []
        return int(f())
    except (OSError, AttributeError, ValueError):
        return None


LANG: str = pick_lang(_windows_ui_langid())


def t(key: str, **kw) -> str:
    """The words of `key` in the current language, with `kw` filled in.  An unknown key raises KeyError."""
    row = TABLE[key]
    text = row.get(LANG) or row["en"]
    return text.format(**kw) if kw else text
