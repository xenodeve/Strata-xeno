"""Strata Desktop's tray icon (#310 part 3): the controller's TrayPort on pystray.  `pystray` and `PIL` are imported
inside functions only (the unit tests and `python -m pytest desktop` with the global python never load them).

The icon is desktop/ui/icon.png in one colour: `state.icon_color` (green online; amber connecting, pending or
offline; red refused or error; grey not connected).  The tooltip is `state.tooltip`, the menu `state.menu`: Open
(the default: a click on the icon), Copy code (only while the owner's approval is awaited), Switch server, Quit.

Threads: the icon's loop runs `Icon.run` on the thread `desktop-tray` (on Windows that is exactly what
`Icon.run_detached` does, but its thread is unnamed and not handed back, so it could not be joined; Task 1 found it
non-daemon).  A menu item never runs on that thread: each action runs on its own short thread `desktop-tray-action`,
so a Quit (which stops this icon and joins `desktop-tray`) or a connect that waits never blocks the icon's messages.
`stop` waits for the icon to be up, stops it and joins its thread, so Quit leaves no icon and no thread behind.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable

from desktop import state, words
from desktop.state import TrayState

log = logging.getLogger("strata.desktop.tray")

ICON = Path(__file__).resolve().parent / "ui" / "icon.png"
ICON_SIZE = 64
COLORS = {"green": (31, 138, 76), "amber": (194, 124, 14), "red": (179, 38, 30), "grey": (138, 146, 155)}
READY_S = 5.0
JOIN_S = 5.0
TITLE_MOST = 63                 # NOTIFYICONDATAW szInfoTitle
TEXT_MOST = 255                 # szInfo


def tinted(color: str, path: Path = ICON, size: int = ICON_SIZE):
    """The icon's shape in `color` (one of COLORS): its alpha kept, every pixel that colour."""
    from PIL import Image
    with Image.open(path) as im:
        base = im.convert("RGBA").resize((size, size), Image.Resampling.LANCZOS)
    out = Image.new("RGBA", base.size, COLORS.get(color, COLORS["grey"]) + (255,))
    out.putalpha(base.getchannel("A"))
    return out


class Tray:
    def __init__(self, on_open: Callable[[], None], on_switch: Callable[[], None],
                 on_copy_code: Callable[[], object], on_quit: Callable[[], None]):
        self._actions = {"open": on_open, "switch": on_switch, "copy_code": on_copy_code, "quit": on_quit}
        self._state = TrayState("idle")
        self._lock = threading.Lock()
        self._icon = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stopped = False
        self._images: dict[str, object] = {}

    def __repr__(self) -> str:
        return f"Tray({self._state.kind!r})"

    def _image(self, color: str):
        if color not in self._images:
            self._images[color] = tinted(color)
        return self._images[color]

    def _act(self, name: str) -> Callable[..., None]:
        def run(icon=None, item=None) -> None:
            def call():
                try:
                    self._actions[name]()
                except Exception as e:  # noqa: BLE001
                    log.error("the tray's %s raised %s", name, type(e).__name__, exc_info=True)
            threading.Thread(target=call, name="desktop-tray-action", daemon=True).start()
        return run

    def _shown(self, name: str) -> Callable[..., bool]:
        return lambda item: name in state.menu(self._state)

    def start(self) -> None:
        import pystray
        with self._lock:
            if self._icon is not None or self._stopped:
                return
            item = pystray.MenuItem
            menu = pystray.Menu(
                item(words.t("menu_open"), self._act("open"), default=True),
                item(words.t("menu_copy_code"), self._act("copy_code"), visible=self._shown("copy_code")),
                item(words.t("menu_switch"), self._act("switch")),
                item(words.t("menu_quit"), self._act("quit")))
            s = self._state
            self._icon = pystray.Icon("StrataDesktop", self._image(state.icon_color(s)), state.tooltip(s), menu=menu)

            def setup(icon) -> None:
                icon.visible = True
                self._ready.set()

            self._thread = threading.Thread(target=self._icon.run, args=(setup,), name="desktop-tray", daemon=True)
            self._thread.start()

    def update(self, s: TrayState) -> None:
        with self._lock:
            self._state = s
            icon = self._icon if not self._stopped else None
        if icon is None:
            return
        try:
            icon.icon = self._image(state.icon_color(s))
            icon.title = state.tooltip(s)
            icon.update_menu()
        except Exception as e:  # noqa: BLE001
            log.warning("updating the tray icon raised %s", type(e).__name__)

    def notify(self, title: str, text: str) -> None:
        with self._lock:
            icon = self._icon if not self._stopped else None
        if icon is None or not self._ready.is_set():
            return
        try:
            icon.notify(text[:TEXT_MOST], title[:TITLE_MOST])
        except Exception as e:  # noqa: BLE001
            log.warning("the tray notice raised %s", type(e).__name__)

    def stop(self) -> None:
        """The icon off the notification area and its thread ended (joined unless called from it).  Twice is fine."""
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            icon, thread = self._icon, self._thread
        if icon is None:
            return
        if not self._ready.wait(READY_S):              # pystray's stop does nothing before the loop is up
            log.warning("the tray icon was not up after %g s", READY_S)
        try:
            icon.stop()
        except Exception as e:  # noqa: BLE001
            log.warning("stopping the tray icon raised %s", type(e).__name__)
        if thread is not None and thread is not threading.current_thread():
            thread.join(JOIN_S)
            if thread.is_alive():
                log.warning("the tray's thread did not end in %g s", JOIN_S)
