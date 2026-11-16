"""The tray's states (#310 part 3): one frozen TrayState, `step` for each event the runner host reports, and what the
tray shows for a state (its tooltip, its icon's colour, its menu).  No GUI import: desktop/gui.py draws it.

Events (desktop/runner_host.py, serve/runner.py RunnerLoop(on_state=)): "connecting", "online" {"id", "who"[,
"no_shell"]}, "pending" {"code"}, "offline" {"message"}, "refused" {"kind", "message"}, "error" {"message"}, "stopped".
"refused" and "error" are end states: only a new "connecting" leaves them.  `server` (the server's name in the list) is
set by the caller when it makes the state, or by "connecting" {"server"}; every other event keeps it.
"""
from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from typing import Literal

from desktop import words

Kind = Literal["idle", "connecting", "online", "pending", "offline", "refused", "error"]
KINDS: tuple[str, ...] = ("idle", "connecting", "online", "pending", "offline", "refused", "error")
TOOLTIP_MAX = 127               # Windows' notification-area tip: 128 characters with the closing NUL
ENDS = ("refused", "error")     # end states: later events but "connecting" are ignored
REFUSED_WORDS = {"key": "tray_refused_key", "device_refused": "tray_refused_device",
                 "runner_replaced": "tray_refused_replaced"}
PROTOCOL = re.compile(r"protocol (\d+)")


@dataclass(frozen=True)
class TrayState:
    kind: Kind
    server: str = ""
    who: str = ""
    runner_id: str = ""
    code: str = ""
    refused: str = ""
    message: str = ""
    no_shell: bool = False


def _text(v) -> str:
    return v if isinstance(v, str) else "" if v is None else str(v)


def step(s: TrayState, event: str, data: dict) -> TrayState:
    """The state after `event`.  An unknown event, or any event but "connecting" in an end state, changes nothing."""
    data = data if isinstance(data, dict) else {}
    if event == "connecting":
        return TrayState("connecting", server=_text(data.get("server", s.server)))
    if s.kind in ENDS:
        return s
    if event == "online":
        return dataclasses.replace(s, kind="online", who=_text(data.get("who")), runner_id=_text(data.get("id")),
                                   code="", refused="", message="", no_shell=bool(data.get("no_shell", s.no_shell)))
    if event == "pending":
        return dataclasses.replace(s, kind="pending", code=_text(data.get("code")), message="")
    if event == "offline":
        return dataclasses.replace(s, kind="offline", message=_text(data.get("message")))
    if event == "refused":
        return dataclasses.replace(s, kind="refused", refused=_text(data.get("kind")) or "refused",
                                   message=_text(data.get("message")), code="")
    if event == "error":
        return dataclasses.replace(s, kind="error", message=_text(data.get("message")), code="")
    if event == "stopped":
        return TrayState("idle", server=s.server)
    return s


def _render(s: TrayState, server: str) -> str:
    if s.kind == "connecting":
        text = words.t("tray_connecting", server=server)
    elif s.kind == "online":
        text = words.t("tray_online", server=server, who=s.who)
    elif s.kind == "pending":
        text = words.t("pending", server=server, code=s.code)
    elif s.kind == "offline":
        text = words.t("tray_offline", server=server)
    elif s.kind == "refused":
        if s.refused == "runner_protocol":
            m = PROTOCOL.search(s.message)
            text = words.t("update", n=m.group(1) if m else "?", server=server)
        else:
            text = words.t(REFUSED_WORDS.get(s.refused, "tray_refused"), server=server)
    elif s.kind == "error":
        text = words.t("tray_error", message=s.message)
    else:
        text = words.t("tray_idle")
    if s.no_shell and s.kind in ("online", "pending", "offline"):   # the runner works, but without Bash
        text += words.t("tray_no_shell")
    return text


def tooltip(s: TrayState) -> str:
    """The tray icon's tooltip, at most TOOLTIP_MAX characters: the server's name is cut first, then the end."""
    text = _render(s, s.server)
    over = len(text) - TOOLTIP_MAX
    if over > 0 and s.server and s.server in text:
        keep = max(0, len(s.server) - over - 1)
        text = _render(s, s.server[:keep] + "…")
    if len(text) > TOOLTIP_MAX:
        text = text[:TOOLTIP_MAX - 1] + "…"
    return text


def icon_color(s: TrayState) -> str:
    """green online; amber while it connects, waits for approval or cannot reach the server; red when refused or the
    tools could not start; grey when not connected."""
    if s.kind == "online":
        return "green"
    if s.kind in ("connecting", "pending", "offline"):
        return "amber"
    if s.kind in ENDS:
        return "red"
    return "grey"


def menu(s: TrayState) -> list[str]:
    """The tray menu's items (desktop/words.py "menu_<item>"): "copy_code" only while the owner's approval is awaited."""
    if s.kind == "pending":
        return ["open", "copy_code", "switch", "quit"]
    return ["open", "switch", "quit"]
