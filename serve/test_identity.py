"""serve/test_identity.py - named keys, roles and device approval (serve/identity.py), no server and no engine.

    python -m unittest serve.test_identity -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import identity  # noqa: E402
from serve.identity import LOCAL_GUEST, OWNER, Identity, KeyStore, visible_record  # noqa: E402

ANN = Identity("ann", "guest")
BOB = Identity("bob", "guest")


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = os.path.join(self._tmp.name, "strata-keys.json")

    def store(self, legacy=""):
        return KeyStore(self.path, legacy)

    def read(self):
        return json.loads(Path(self.path).read_text(encoding="utf-8"))


class Keys(Base):
    def test_legacy_key_is_owner(self):
        s = KeyStore(None, "abc")
        self.assertEqual(s.match("abc"), OWNER)
        self.assertIsNone(s.match("x"))
        self.assertIs(OWNER.is_owner, True)
        self.assertIs(LOCAL_GUEST.is_owner, False)

    def test_legacy_keys_as_list_and_comma_form(self):
        self.assertEqual(KeyStore(None, ["k1", "k,2"]).match("k,2"), OWNER)
        self.assertEqual(KeyStore(None, "k1, k2").match("k2"), OWNER)
        self.assertIsNone(KeyStore(None, "").match(""))

    def test_legacy_key_with_non_ascii_matches_in_both_header_forms(self):
        s = KeyStore(None, "kéy")
        self.assertEqual(s.match("kéy".encode("utf-8").decode("latin-1")), OWNER)    # as http.server reads UTF-8
        self.assertEqual(s.match("kéy".encode("latin-1").decode("latin-1")), OWNER)  # a client that sent Latin-1

    def test_created_key_matches_and_is_stored_hashed(self):
        s = self.store()
        k = s.create("ann", "guest")
        self.assertTrue(k.startswith("sk-strata-"))
        self.assertEqual(s.match(k), ANN)
        self.assertNotIn(k, Path(self.path).read_text(encoding="utf-8"))
        self.assertEqual(self.read()["keys"][0]["name"], "ann")
        self.assertEqual(len(self.read()["keys"][0]["sha256"]), 64)
        self.assertIsNone(s.match(k + "x"))
        self.assertEqual(self.store().match(k), ANN)                   # a fresh start reads the file

    def test_a_non_ascii_key_in_a_hand_written_hash_matches_in_both_forms(self):
        import hashlib
        Path(self.path).write_text(json.dumps({"keys": [{
            "name": "ann", "role": "guest", "created": 1, "devices": {},
            "sha256": hashlib.sha256("kéy".encode("utf-8")).hexdigest()}]}), encoding="utf-8")
        s = self.store()
        self.assertEqual(s.match("kéy".encode("utf-8").decode("latin-1")), ANN)
        self.assertEqual(s.match("kéy"), ANN)

    def test_two_created_keys_differ_and_both_match(self):
        s = self.store()
        a, b = s.create("ann", "guest"), s.create("bob", "owner")
        self.assertNotEqual(a, b)
        self.assertEqual(s.match(a), ANN)
        self.assertEqual(s.match(b), Identity("bob", "owner"))

    def test_names_are_unique_and_roles_checked(self):
        s = self.store()
        s.create("ann", "guest")
        with self.assertRaises(ValueError):
            s.create("ann", "guest")
        with self.assertRaises(ValueError):
            s.create("Ann", "guest")                                   # unique ignoring case
        with self.assertRaises(ValueError):
            s.create("bob", "admin")
        for bad in ("", "   ", "x" * 65, "a\nb"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                s.create(bad, "guest")

    def test_owner_and_anonymous_are_reserved(self):
        s = self.store()
        for name in ("owner", "Owner", "anonymous"):
            with self.assertRaises(ValueError, msg=name):
                s.create(name, "guest")
        s.create("ann", "guest")
        for name in ("owner", "anonymous"):
            with self.assertRaises(ValueError, msg=name):
                s.rename("ann", name)

    def test_the_legacy_owner_cannot_be_revoked_renamed_or_demoted(self):
        s = self.store("abc")
        with self.assertRaises(ValueError):
            s.revoke("owner")
        with self.assertRaises(ValueError):
            s.rename("owner", "boss")
        with self.assertRaises(ValueError):
            s.set_role("owner", "guest")
        self.assertEqual(s.match("abc"), OWNER)

    def test_revoke_removes_the_key(self):
        s = self.store()
        k = s.create("ann", "guest")
        self.assertIs(s.revoke("ann"), True)
        self.assertIsNone(s.match(k))
        self.assertIs(s.revoke("ann"), False)
        self.assertIsNone(self.store().match(k))

    def test_rename_and_set_role_take_effect_at_once(self):
        s = self.store()
        k = s.create("ann", "guest")
        self.assertIs(s.rename("ann", "anna"), True)
        self.assertEqual(s.match(k), Identity("anna", "guest"))
        self.assertIs(s.set_role("anna", "owner"), True)
        self.assertEqual(s.match(k), Identity("anna", "owner"))
        with self.assertRaises(ValueError):
            s.set_role("anna", "admin")
        self.assertIs(s.rename("nobody", "x"), False)
        self.assertIs(s.set_role("nobody", "guest"), False)
        s.create("bob", "guest")
        with self.assertRaises(ValueError):
            s.rename("bob", "anna")

    def test_view_never_holds_a_key_or_a_hash(self):
        s = self.store()
        k = s.create("ann", "guest")
        s.device(ANN, "d1", "PC-1", "26.1.1.1", 100.0)
        rows = s.view()
        self.assertEqual([r["name"] for r in rows], ["ann"])
        self.assertEqual(rows[0]["role"], "guest")
        self.assertIn("created", rows[0])
        self.assertEqual(rows[0]["devices"][0]["device"], "d1")
        text = json.dumps(rows)
        self.assertNotIn(k, text)
        self.assertNotIn(self.read()["keys"][0]["sha256"], text)
        self.assertNotIn("sha256", text)

    def test_any_key(self):
        self.assertIs(KeyStore(None, "").any_key(), False)
        self.assertIs(KeyStore(None, "a").any_key(), True)
        s = self.store()
        self.assertIs(s.any_key(), False)
        s.create("ann", "guest")
        self.assertIs(s.any_key(), True)

    def test_team_mode_is_the_keys_file(self):
        self.assertIs(KeyStore(None, "abc").team_mode(), False)
        s = self.store("abc")
        self.assertIs(s.team_mode(), False)                            # no file yet: the single owner key as today
        s.create("ann", "guest")
        self.assertTrue(os.path.exists(self.path))
        self.assertIs(s.team_mode(), True)
        self.assertIs(self.store("abc").team_mode(), True)

    def test_a_file_made_by_hand_turns_team_mode_on(self):
        s = self.store()
        self.assertIs(s.team_mode(), False)
        Path(self.path).write_text(json.dumps({"keys": []}), encoding="utf-8")
        self.assertIs(s.team_mode(), True)

    def test_log_is_capped_and_never_holds_a_key(self):
        s = self.store()
        k = s.create("ann", "guest")
        for i in range(520):
            s.rename("ann", "n%d" % i) if i % 2 == 0 else s.rename("n%d" % (i - 1), "ann")
        log = self.read()["log"]
        self.assertEqual(len(log), 500)
        self.assertNotIn(k, json.dumps(log))
        self.assertEqual(set(log[-1]), {"t", "what", "key", "device", "by"})

    def test_the_file_is_replaced_not_left_half_written(self):
        s = self.store()
        s.create("ann", "guest")
        self.assertEqual(sorted(os.listdir(self._tmp.name)), ["strata-keys.json"])     # no temp file left behind


class Reload(Base):
    def test_reload_after_external_edit(self):
        s = self.store()
        s.create("ann", "guest")
        import hashlib
        doc = self.read()
        doc["keys"].append({"name": "bob", "role": "guest", "created": 1, "devices": {},
                            "sha256": hashlib.sha256(b"sk-by-hand").hexdigest()})
        Path(self.path).write_text(json.dumps(doc), encoding="utf-8")
        st = os.stat(self.path)
        os.utime(self.path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))      # a later mtime
        s.reload_if_changed()
        self.assertEqual(s.match("sk-by-hand"), BOB)

    def test_match_reloads_by_itself_but_at_most_once_a_second(self):
        s = self.store()
        s.create("ann", "guest")
        import hashlib
        now = [1000.0]
        with mock.patch.object(identity, "_clock", lambda: now[0]):
            s.match("x")                                               # looks at the file: nothing new
            doc = self.read()
            doc["keys"].append({"name": "bob", "role": "guest", "created": 1, "devices": {},
                                "sha256": hashlib.sha256(b"sk-by-hand").hexdigest()})
            Path(self.path).write_text(json.dumps(doc), encoding="utf-8")
            st = os.stat(self.path)
            os.utime(self.path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
            now[0] += 0.2
            self.assertIsNone(s.match("sk-by-hand"))                   # the same second does not look again
            now[0] += 1.0
            self.assertEqual(s.match("sk-by-hand"), BOB)

    def test_broken_file_keeps_last_good_keys(self):
        s = self.store()
        k = s.create("ann", "guest")
        Path(self.path).write_text("{", encoding="utf-8")
        st = os.stat(self.path)
        os.utime(self.path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
        with mock.patch("builtins.print") as log:
            s.reload_if_changed()
            s.reload_if_changed()
        self.assertEqual(log.call_count, 1)                            # one line, not one per check
        self.assertEqual(s.match(k), ANN)
        self.assertIsNone(s.match("a-random-string"))
        self.assertIs(s.any_key(), True)

    def test_a_file_of_the_wrong_shape_is_a_broken_file(self):
        s = self.store()
        k = s.create("ann", "guest")
        for text in ("[]", '{"keys": 5}', "null", ""):
            Path(self.path).write_text(text, encoding="utf-8")
            st = os.stat(self.path)
            os.utime(self.path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
            with mock.patch("builtins.print"):
                s.reload_if_changed()
            self.assertEqual(s.match(k), ANN, text)

    def test_a_broken_file_at_start_never_opens_the_server(self):
        Path(self.path).write_text("{", encoding="utf-8")
        s = self.store()
        self.assertIsNone(s.match("anything"))
        self.assertIs(s.any_key(), True)                               # the file exists: keys are required
        self.assertIs(s.team_mode(), True)

    def test_an_entry_with_an_unknown_role_never_matches(self):
        import hashlib
        Path(self.path).write_text(json.dumps({"keys": [{
            "name": "x", "role": "admin", "created": 1, "devices": {},
            "sha256": hashlib.sha256(b"k").hexdigest()}]}), encoding="utf-8")
        self.assertIsNone(self.store().match("k"))

    def test_a_write_after_an_outside_edit_keeps_the_edit(self):
        s = self.store()
        s.create("ann", "guest")
        import hashlib
        doc = self.read()
        doc["keys"].append({"name": "bob", "role": "guest", "created": 1, "devices": {},
                            "sha256": hashlib.sha256(b"sk-by-hand").hexdigest()})
        Path(self.path).write_text(json.dumps(doc), encoding="utf-8")
        st = os.stat(self.path)
        os.utime(self.path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
        s.create("cy", "guest")                                        # no match() in between
        self.assertEqual([k["name"] for k in self.read()["keys"]], ["ann", "bob", "cy"])

    def _break(self, text="{ typo"):
        Path(self.path).write_text(text, encoding="utf-8")
        st = os.stat(self.path)
        os.utime(self.path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))

    def test_no_write_while_the_file_is_broken(self):
        s = self.store("abc")
        k = s.create("ann", "guest")
        s.device(ANN, "d1", "PC", "1.1.1.1", 1000.0)
        s.decide_device("ann", "d1", True, OWNER)
        self._break()
        with mock.patch("builtins.print"):
            s.reload_if_changed()
            for call in (lambda: s.create("bob", "guest"), lambda: s.rename("ann", "anna"),
                         lambda: s.set_role("ann", "owner"), lambda: s.revoke("ann"),
                         lambda: s.decide_device("ann", "d1", False, OWNER),
                         lambda: s.clear_device("ann", "d1")):
                with self.assertRaises(identity.KeysFileBad):
                    call()
            self.assertEqual(s.device(OWNER, "o1", "Desk", "2.2.2.2", 1000.0), "pending")     # the legacy owner's first device
            self.assertEqual(s.device(ANN, "d1", "PC", "3.3.3.3", 5000.0), "approved")        # a last_seen / address change
            self.assertEqual(s.device(ANN, "d9", "PC", "3.3.3.3", 5000.0), "pending")         # a new device
        self.assertEqual(Path(self.path).read_text(encoding="utf-8"), "{ typo")               # the broken bytes are still there
        self.assertEqual(s.match(k), ANN)                                                     # the last good keys still answer
        self.assertIsInstance(identity.KeysFileBad("x"), ValueError)

    def test_writes_resume_when_the_file_is_fixed(self):
        s = self.store()
        s.create("ann", "guest")
        good = Path(self.path).read_text(encoding="utf-8")
        self._break()
        with mock.patch("builtins.print"):
            s.reload_if_changed()
        with self.assertRaises(identity.KeysFileBad):
            s.create("bob", "guest")
        Path(self.path).write_text(good, encoding="utf-8")
        st = os.stat(self.path)
        os.utime(self.path, ns=(st.st_atime_ns, st.st_mtime_ns + 9_000_000_000))
        s.create("bob", "guest")
        self.assertEqual([k["name"] for k in self.read()["keys"]], ["ann", "bob"])

    def test_deleting_a_broken_file_lets_a_new_one_be_written(self):
        s = self.store()
        s.create("ann", "guest")
        self._break()
        with mock.patch("builtins.print"):
            s.reload_if_changed()
        os.unlink(self.path)
        s.reload_if_changed()
        s.create("bob", "guest")
        self.assertEqual([k["name"] for k in self.read()["keys"]], ["ann", "bob"])    # the held keys are written back

    def test_a_file_broken_at_start_is_not_overwritten(self):
        Path(self.path).write_text("{ typo", encoding="utf-8")
        with mock.patch("builtins.print"):
            s = self.store()
            with self.assertRaises(identity.KeysFileBad):
                s.create("ann", "guest")
        self.assertEqual(Path(self.path).read_text(encoding="utf-8"), "{ typo")

    def test_reserved_names_in_the_file_are_dropped(self):
        import hashlib
        h = lambda k: hashlib.sha256(k).hexdigest()
        Path(self.path).write_text(json.dumps({"keys": [
            {"name": "owner", "role": "guest", "created": 1, "devices": {}, "sha256": h(b"k1")},
            {"name": "Anonymous", "role": "owner", "created": 1, "devices": {}, "sha256": h(b"k2")},
            {"name": "ann", "role": "guest", "created": 1, "devices": {}, "sha256": h(b"k3")}]}), encoding="utf-8")
        with mock.patch("builtins.print") as log:
            s = self.store("legacy")
        self.assertEqual(log.call_count, 2)
        self.assertIsNone(s.match("k1"))
        self.assertIsNone(s.match("k2"))
        self.assertEqual(s.match("k3"), ANN)
        self.assertEqual(s.match("legacy"), OWNER)
        self.assertEqual([r["name"] for r in s.view()], ["ann"])
        self.assertIs(s.may_approve(Identity("owner", "guest"), "owner", "d1"), False)

    def test_the_legacy_entry_in_the_file_is_still_kept(self):
        s = self.store("abc")
        s.device(OWNER, "o1", "", "", 1.0)
        self.assertIn("owner", [r["name"] for r in self.store("abc").view()])

    def test_a_failed_write_leaves_the_store_as_it_was(self):
        s = self.store()
        s.create("ann", "guest")
        with mock.patch("serve.identity.os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                s.create("bob", "guest")
        self.assertEqual([r["name"] for r in s.view()], ["ann"])
        s.create("bob", "guest")                                       # and it still works afterwards


class Devices(Base):
    def setUp(self):
        super().setUp()
        self.s = self.store()
        self.key = self.s.create("ann", "guest")

    def test_new_device_is_pending_then_approved(self):
        s = self.s
        self.assertEqual(s.device(ANN, "d1", "PC-1", "26.1.1.1", 100.0), "pending")
        rows = s.pending()
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]["code"]), 4)
        self.assertEqual({k: rows[0][k] for k in ("key", "device", "label", "addr", "first_seen")},
                         {"key": "ann", "device": "d1", "label": "PC-1", "addr": "26.1.1.1", "first_seen": 100.0})
        self.assertEqual(s.device(ANN, "d1", "PC-1", "26.1.1.1", 101.0), "pending")     # the same device, same code
        self.assertEqual(s.pending()[0]["code"], rows[0]["code"])
        self.assertIs(s.decide_device("ann", "d1", True, OWNER), True)
        self.assertEqual(s.device(ANN, "d1", "PC-1", "26.1.1.1", 102.0), "approved")
        self.assertEqual(s.pending(), [])
        d = self.read()["keys"][0]["devices"]["d1"]
        self.assertEqual((d["status"], d["decided_by"]), ("approved", "owner"))
        self.assertEqual(self.read()["log"][-1]["what"], "approved")

    def test_the_code_is_four_of_a_to_z_and_digits(self):
        for i in range(20):
            self.s.device(ANN, "d%d" % i, "", "", 1.0)
        for row in self.s.pending():
            self.assertRegex(row["code"], r"^[A-Z0-9]{4}$")

    def test_refused_device_stays_refused_until_cleared(self):
        s = self.s
        s.device(ANN, "d1", "PC-1", "26.1.1.1", 100.0)
        self.assertIs(s.decide_device("ann", "d1", False, OWNER), True)
        self.assertEqual(s.device(ANN, "d1", "PC-1", "26.1.1.1", 101.0), "refused")
        self.assertEqual(s.device(ANN, "d1", "PC-1", "26.1.1.1", 102.0), "refused")
        self.assertEqual(s.pending(), [])
        self.assertIs(s.clear_device("ann", "d1"), True)
        self.assertEqual(s.device(ANN, "d1", "PC-1", "26.1.1.1", 103.0), "pending")
        self.assertIs(s.clear_device("ann", "nope"), False)
        self.assertIs(s.clear_device("nobody", "d1"), False)

    def test_a_guest_cannot_undo_the_owners_refusal(self):
        s = self.s
        s.device(ANN, "d1", "", "", 1.0)
        s.decide_device("ann", "d1", True, OWNER)
        s.device(ANN, "d2", "", "", 2.0)
        s.decide_device("ann", "d2", False, OWNER)
        self.assertIs(s.decide_device("ann", "d2", True, ANN), False)
        self.assertEqual(s.device(ANN, "d2", "", "", 3.0), "refused")
        self.assertIs(s.decide_device("ann", "d2", True, OWNER), True)        # the owner may change their mind

    def test_decide_on_an_unknown_device_is_false(self):
        self.assertIs(self.s.decide_device("ann", "nope", True, OWNER), False)
        self.assertIs(self.s.decide_device("nobody", "d1", True, OWNER), False)

    def test_may_approve_rules(self):
        s = self.s
        s.create("bob", "guest")
        self.assertIs(s.may_approve(OWNER, "ann", None), True)
        self.assertIs(s.may_approve(OWNER, "bob", "d9"), True)
        self.assertIs(s.may_approve(ANN, "ann", "d1"), False)                  # her own key, but no approved device yet
        s.device(ANN, "d1", "", "", 1.0)
        self.assertIs(s.may_approve(ANN, "ann", "d1"), False)                  # d1 is still pending
        s.decide_device("ann", "d1", True, OWNER)
        self.assertIs(s.may_approve(ANN, "ann", "d1"), True)                   # from an approved device of her key
        self.assertIs(s.may_approve(ANN, "bob", "d1"), False)                  # never for another key
        s.device(ANN, "d2", "", "", 2.0)
        self.assertIs(s.may_approve(ANN, "ann", "d2"), False)                  # from a pending device
        self.assertIs(s.may_approve(ANN, "ann", None), False)
        self.assertIs(s.may_approve(ANN, "ann", "unknown"), False)
        self.assertIs(s.may_approve(LOCAL_GUEST, "ann", "d1"), False)

    def test_a_refused_device_cannot_approve(self):
        s = self.s
        s.device(ANN, "d1", "", "", 1.0)
        s.decide_device("ann", "d1", False, OWNER)
        self.assertIs(s.may_approve(ANN, "ann", "d1"), False)

    def test_an_identity_without_a_key_is_approved_and_nothing_is_recorded(self):
        before = Path(self.path).read_text(encoding="utf-8")
        self.assertEqual(self.s.device(LOCAL_GUEST, "d1", "x", "127.0.0.1", 1.0), "approved")
        self.assertEqual(Path(self.path).read_text(encoding="utf-8"), before)
        self.assertEqual(self.s.pending(), [])

    def test_no_entry_means_approved_whatever_the_device_id(self):
        for bad in ("", None, "x" * 500):
            self.assertEqual(self.s.device(LOCAL_GUEST, bad, "", "127.0.0.1", 1.0), "approved")
        self.assertEqual(self.s.device(Identity("ghost", "guest"), "", "", "", 1.0), "approved")
        self.assertEqual(self.s.device(ANN, "", "", "", 1.0), "refused")            # a key's own bad id stays refused

    def test_a_store_without_a_file_path_records_nothing(self):
        s = KeyStore(None, "abc")
        self.assertEqual(s.device(OWNER, "d1", "x", "1.1.1.1", 1.0), "approved")
        self.assertEqual(s.pending(), [])

    def test_the_legacy_owner_gets_a_reserved_entry_on_demand(self):
        s = self.store("abc")
        self.assertNotIn("owner", [k["name"] for k in self.read()["keys"]])
        self.assertEqual(s.device(OWNER, "o1", "Desk", "26.0.0.2", 5.0), "pending")
        entry = [k for k in self.read()["keys"] if k["name"] == "owner"][0]
        self.assertEqual((entry["legacy"], entry["role"]), (True, "owner"))
        self.assertNotIn("sha256", entry)
        self.assertEqual(s.pending()[0]["key"], "owner")
        s.decide_device("owner", "o1", True, OWNER)
        self.assertEqual(self.store("abc").device(OWNER, "o1", "Desk", "26.0.0.2", 6.0), "approved")
        self.assertIn("owner", [r["name"] for r in s.view()])
        self.assertIs(s.may_approve(OWNER, "owner", None), True)

    def test_a_legacy_entry_in_the_file_matches_no_key(self):
        s = self.store("abc")
        s.device(OWNER, "o1", "", "", 1.0)
        self.assertIsNone(s.match(""))
        self.assertIsNone(s.match("owner"))
        self.assertEqual(s.match("abc"), OWNER)

    def test_revoking_a_key_drops_its_devices(self):
        self.s.device(ANN, "d1", "", "", 1.0)
        self.s.revoke("ann")
        self.assertEqual(self.s.pending(), [])

    def test_bad_device_ids_are_refused_and_not_recorded(self):
        for bad in ("", "x" * 200, None):
            self.assertEqual(self.s.device(ANN, bad, "", "", 1.0), "refused")
        self.assertEqual(self.s.pending(), [])

    def test_a_key_holds_at_most_a_hundred_devices(self):
        for i in range(identity.MAX_DEVICES):
            self.assertEqual(self.s.device(ANN, "d%d" % i, "", "", 1.0), "pending")
        self.assertEqual(self.s.device(ANN, "one-more", "", "", 1.0), "refused")
        self.assertEqual(self.s.device(ANN, "d0", "", "", 2.0), "pending")             # a known one still answers

    def test_label_and_address_are_cut_to_length(self):
        self.s.device(ANN, "d1", "L" * 500, "A" * 500, 1.0)
        row = self.s.pending()[0]
        self.assertLessEqual(len(row["label"]), 80)
        self.assertLessEqual(len(row["addr"]), 80)

    def test_last_seen_is_written_at_most_once_a_minute(self):
        s = self.s
        s.device(ANN, "d1", "", "1.1.1.1", 1000.0)
        s.decide_device("ann", "d1", True, OWNER)
        s.device(ANN, "d1", "", "1.1.1.1", 1010.0)
        self.assertEqual(self.read()["keys"][0]["devices"]["d1"]["last_seen"], 1000.0)   # not rewritten for 10 s
        s.device(ANN, "d1", "", "1.1.1.1", 1100.0)
        self.assertEqual(self.read()["keys"][0]["devices"]["d1"]["last_seen"], 1100.0)
        s.device(ANN, "d1", "", "2.2.2.2", 1101.0)                                       # a new address is written at once
        self.assertEqual(self.read()["keys"][0]["devices"]["d1"]["addr"], "2.2.2.2")


class Visible(unittest.TestCase):
    RECORD = {"id": "r1", "who": "bob", "state": "running", "queue_s": 1.5, "preview": "secret prompt",
              "input": "x", "output": "y", "reasoning": "z", "response": "r", "prompt": "p", "messages": [1],
              "detail": {"rounds": []}, "has_detail": True, "detail_deleted": False,
              "prompt_tokens": 12, "timings": {"decode_ms": 9}}

    def test_visible_record(self):
        rec = self.RECORD
        self.assertEqual(visible_record(rec, OWNER), rec)
        self.assertEqual(visible_record(rec, Identity("bob", "guest")), rec)
        own = dict(rec, who="ann", preview="mine")
        self.assertEqual(visible_record(own, ANN)["preview"], "mine")
        other = visible_record(rec, ANN)
        self.assertNotIn("preview", other)
        for gone in ("input", "output", "reasoning", "response", "prompt", "messages", "detail", "has_detail",
                     "detail_deleted"):
            self.assertNotIn(gone, other)
        self.assertEqual((other["state"], other["who"], other["queue_s"]), ("running", "bob", 1.5))
        self.assertEqual((other["prompt_tokens"], other["timings"]), (12, {"decode_ms": 9}))
        self.assertIn("preview", rec)                                  # the stored record is not changed

    def test_a_record_without_who_is_hidden_from_a_guest(self):
        self.assertNotIn("preview", visible_record({"preview": "p", "state": "done"}, ANN))
        self.assertNotIn("preview", visible_record({"preview": "p"}, LOCAL_GUEST))


if __name__ == "__main__":
    unittest.main()
