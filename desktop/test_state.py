"""desktop/test_state.py - the tray's states (#310 part 3 task 3): each event's transition, refused and error as end
states, the tooltip's words and its 127-character limit, the icon's colour and the menu.

    python -m pytest desktop/test_state.py -q
"""
from __future__ import annotations

import unittest

from desktop import state, words
from desktop.state import TrayState, icon_color, menu, step, tooltip

SERVER = "team-box"


class English(unittest.TestCase):
    def setUp(self):
        self._lang = words.LANG
        words.LANG = "en"                                       # a Thai display would otherwise pick the Thai rows
        self.addCleanup(setattr, words, "LANG", self._lang)


class Transitions(English):
    def test_starts_idle(self):
        s = TrayState("idle", server=SERVER)
        self.assertEqual(tooltip(s), "Strata Desktop - not connected")
        self.assertEqual(icon_color(s), "grey")

    def test_connecting(self):
        s = step(TrayState("idle", server=SERVER), "connecting", {})
        self.assertEqual((s.kind, s.server), ("connecting", SERVER))
        self.assertEqual(tooltip(s), "Connecting to team-box…")
        self.assertEqual(icon_color(s), "amber")

    def test_connecting_names_a_new_server(self):
        s = step(TrayState("online", server=SERVER, who="ann", runner_id="r1"), "connecting", {"server": "other"})
        self.assertEqual((s.kind, s.server, s.who, s.runner_id), ("connecting", "other", "", ""))

    def test_online(self):
        s = step(TrayState("connecting", server=SERVER), "online", {"id": "r1", "who": "ann"})
        self.assertEqual((s.kind, s.who, s.runner_id, s.server), ("online", "ann", "r1", SERVER))
        self.assertEqual(tooltip(s), "Online - team-box as ann")
        self.assertEqual(icon_color(s), "green")

    def test_online_clears_the_code(self):
        s = step(TrayState("pending", server=SERVER, code="K7QZ"), "online", {"id": "r1", "who": "ann"})
        self.assertEqual((s.kind, s.code), ("online", ""))

    def test_pending(self):
        s = step(TrayState("connecting", server=SERVER), "pending", {"code": "K7QZ"})
        self.assertEqual((s.kind, s.code), ("pending", "K7QZ"))
        self.assertEqual(tooltip(s), words.t("pending", server=SERVER, code="K7QZ"))
        self.assertEqual(tooltip(s), "Waiting for approval on team-box: code K7QZ")
        self.assertEqual(icon_color(s), "amber")

    def test_offline_keeps_who_and_id(self):
        s = step(TrayState("online", server=SERVER, who="ann", runner_id="r1"), "offline", {"message": "reset"})
        self.assertEqual((s.kind, s.who, s.runner_id, s.message), ("offline", "ann", "r1", "reset"))
        self.assertEqual(tooltip(s), "Offline - team-box cannot be reached; trying again")
        self.assertEqual(icon_color(s), "amber")

    def test_offline_then_online(self):
        s = step(TrayState("offline", server=SERVER, who="ann", runner_id="r1", message="x"), "online",
                 {"id": "r1", "who": "ann"})
        self.assertEqual((s.kind, s.message), ("online", ""))

    def test_refused(self):
        s = step(TrayState("online", server=SERVER, who="ann"), "refused", {"kind": "key", "message": ""})
        self.assertEqual((s.kind, s.refused), ("refused", "key"))
        self.assertEqual(icon_color(s), "red")

    def test_refused_words_by_kind(self):
        cases = {"key": "The key for team-box was refused (wrong or revoked)",
                 "device_refused": "The owner of team-box refused this PC",
                 "runner_replaced": "Another Strata runner on this PC took over",
                 "refused": "team-box refused the runner",
                 "something_new": "team-box refused the runner"}
        for kind, text in cases.items():
            s = step(TrayState("connecting", server=SERVER), "refused", {"kind": kind, "message": "m"})
            self.assertEqual(tooltip(s), text, kind)

    def test_error(self):
        s = step(TrayState("connecting", server=SERVER), "error", {"message": "runner.json must be a JSON object"})
        self.assertEqual((s.kind, s.message), ("error", "runner.json must be a JSON object"))
        self.assertEqual(tooltip(s), "My PC's tools could not start: runner.json must be a JSON object")
        self.assertEqual(icon_color(s), "red")

    def test_stopped_is_idle(self):
        s = step(TrayState("online", server=SERVER, who="ann", runner_id="r1"), "stopped", {})
        self.assertEqual(s.kind, "idle")
        self.assertEqual(tooltip(s), "Strata Desktop - not connected")

    def test_unknown_event_changes_nothing(self):
        s = TrayState("online", server=SERVER, who="ann", runner_id="r1")
        self.assertEqual(step(s, "something", {"x": 1}), s)

    def test_server_and_no_shell_are_kept(self):
        s = TrayState("connecting", server=SERVER, no_shell=True)
        for ev, data in (("online", {"id": "r1", "who": "ann"}), ("offline", {"message": "m"}),
                         ("pending", {"code": "K"}), ("online", {"id": "r1", "who": "ann"})):
            s = step(s, ev, data)
            self.assertEqual((s.server, s.no_shell), (SERVER, True), ev)

    def test_online_may_say_no_shell(self):
        s = step(TrayState("connecting", server=SERVER), "online", {"id": "r1", "who": "ann", "no_shell": True})
        self.assertTrue(s.no_shell)
        s = step(s, "online", {"id": "r1", "who": "ann", "no_shell": False})
        self.assertFalse(s.no_shell)

    def test_no_shell_words(self):
        s = TrayState("online", server=SERVER, who="ann", no_shell=True)
        self.assertEqual(tooltip(s), "Online - team-box as ann (no shell: Git for Windows was not found, Bash is off)")

    def test_refused_is_terminal(self):
        s = step(TrayState("online", server=SERVER), "refused", {"kind": "runner_replaced", "message": "m"})
        for ev, data in (("online", {"id": "r1", "who": "ann"}), ("offline", {"message": "m"}),
                         ("pending", {"code": "K"}), ("stopped", {}), ("error", {"message": "e"}),
                         ("refused", {"kind": "key", "message": ""})):
            self.assertEqual(step(s, ev, data), s, ev)
        self.assertEqual(step(s, "connecting", {}).kind, "connecting")       # only a new connect leaves it

    def test_error_is_terminal(self):
        s = step(TrayState("connecting", server=SERVER), "error", {"message": "bad"})
        for ev, data in (("online", {"id": "r1", "who": "ann"}), ("stopped", {}), ("refused", {"kind": "key"})):
            self.assertEqual(step(s, ev, data), s, ev)
        self.assertEqual(step(s, "connecting", {}).kind, "connecting")

    def test_state_is_frozen(self):
        with self.assertRaises(Exception):
            TrayState("idle").kind = "online"                   # type: ignore[misc]


class Tooltip(English):
    def test_tooltip_fits_128(self):
        s = TrayState("online", server="s" * 64, who="w" * 64)
        text = tooltip(s)
        self.assertLessEqual(len(text), 127)
        self.assertIn("w" * 64, text)                           # the server name is cut first
        self.assertTrue(text.startswith("Online - sss"), text)

    def test_tooltip_short_is_not_cut(self):
        s = TrayState("online", server="s" * 40, who="w" * 40)
        self.assertEqual(tooltip(s), f"Online - {'s' * 40} as {'w' * 40}")

    def test_tooltip_worst_cases(self):
        """Even with no server name left the text may be too long: it is then cut at the end."""
        long_cases = [TrayState("online", server="s" * 64, who="w" * 64, no_shell=True),
                      TrayState("error", server="s" * 64, message="C:\\" + "p" * 300 + "\\runner.json bad"),
                      TrayState("refused", server="s" * 64, refused="runner_protocol", message="x" * 300),
                      TrayState("pending", server="s" * 64, code="C" * 64, no_shell=True)]
        for lang in ("en", "th"):
            words.LANG = lang
            for s in long_cases:
                self.assertLessEqual(len(tooltip(s)), 127, (lang, s.kind))

    def test_protocol_words(self):
        s = step(TrayState("online", server=SERVER), "refused",
                 {"kind": "runner_protocol", "message": "update Strata Desktop (the server speaks runner protocol 2)"})
        text = tooltip(s)
        self.assertIn("protocol 2", text)
        self.assertEqual(text, words.t("update", n="2", server=SERVER))
        self.assertIn(SERVER, text)

    def test_protocol_words_thai(self):
        words.LANG = "th"
        s = TrayState("refused", server=SERVER, refused="runner_protocol",
                      message="update Strata Desktop (the server speaks runner protocol 2)")
        self.assertEqual(tooltip(s), words.t("update", n="2", server=SERVER))
        self.assertIn("2", tooltip(s))

    def test_protocol_without_a_number(self):
        s = TrayState("refused", server=SERVER, refused="runner_protocol", message="")
        self.assertEqual(tooltip(s), words.t("update", n="?", server=SERVER))

    def test_thai_rows(self):
        words.LANG = "th"
        s = TrayState("online", server=SERVER, who="ann")
        self.assertEqual(tooltip(s), words.t("tray_online", server=SERVER, who="ann"))


class Menu(English):
    def test_menu(self):
        for kind in ("idle", "connecting", "online", "offline", "refused", "error"):
            self.assertEqual(menu(TrayState(kind)), ["open", "switch", "quit"], kind)
        self.assertEqual(menu(TrayState("pending", code="K7QZ")), ["open", "copy_code", "switch", "quit"])

    def test_every_kind_has_a_colour(self):
        for kind in state.KINDS:
            self.assertIn(icon_color(TrayState(kind)), ("green", "amber", "grey", "red"))


if __name__ == "__main__":
    unittest.main()
