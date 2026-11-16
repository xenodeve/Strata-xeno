"""desktop/servers.py: the server list, its tokens kept with DPAPI (#310 part 3 task 1)."""
import base64
import copy
import dataclasses
import json
import os
import pickle

import pytest

from desktop import dpapi, servers
from desktop.servers import Secret, ServerEntry, ServerStore

pytestmark = pytest.mark.skipif(os.name != "nt", reason="the token store needs Windows (DPAPI)")

TOKEN = "sk-strata-Token_0123456789abcdef"


@pytest.fixture
def path(tmp_path):
    return tmp_path / "StrataDesktop" / "servers.json"


def test_save_and_reload(path):
    ServerStore(path).save("Office", "http://26.1.2.3:8095", TOKEN)
    e = ServerStore(path).get("Office")
    assert e == ServerEntry("Office", "http://26.1.2.3:8095", Secret(TOKEN))
    assert e.token.reveal() == TOKEN
    assert [x.name for x in ServerStore(path).entries()] == ["Office"]


def test_file_format(path):
    s = ServerStore(path)
    s.save("Office", "http://26.1.2.3:8095", TOKEN)
    s.save("Home", "https://strata.example.ts.net", None)
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["version"] == 1 and doc["last"] is None and doc["injected"] == []
    assert [set(x) for x in doc["servers"]] == [{"name", "url", "token_dpapi"}] * 2
    blob = base64.b64decode(doc["servers"][0]["token_dpapi"], validate=True)
    assert dpapi.unprotect(blob) == TOKEN.encode("ascii")
    assert doc["servers"][1]["token_dpapi"] is None


def test_file_has_no_token(path):
    ServerStore(path).save("Office", "http://26.1.2.3:8095", TOKEN)
    raw = path.read_bytes()
    assert TOKEN.encode() not in raw
    assert base64.b64encode(TOKEN.encode()) not in raw
    assert base64.b64encode(TOKEN.encode())[:20] not in raw


def test_entry_repr_hides_token(path):
    e = ServerStore(path).save("Office", "http://26.1.2.3:8095", TOKEN)
    for text in (repr(e), str(e), f"{e}", repr(e.token), str(e.token), f"{e.token}", repr([e])):
        assert "***" in text
        assert TOKEN not in text


def test_secret_value_semantics():
    assert Secret("a") == Secret("a") and Secret("a") != Secret("b")
    assert hash(Secret("a")) == hash(Secret("a"))
    assert len({ServerEntry("n", "http://h", Secret("a")), ServerEntry("n", "http://h", Secret("a"))}) == 1
    e = ServerEntry("n", "http://h", Secret("a"))
    assert copy.deepcopy(e) == e and dataclasses.asdict(e)["token"] == Secret("a")
    with pytest.raises(TypeError):
        pickle.dumps(Secret("a"))


def test_url_normalised(path):
    s = ServerStore(path)
    assert s.save("A", "http://26.1.2.3:8095/", TOKEN).url == "http://26.1.2.3:8095"
    assert s.save("B", " https://x.ts.net/strata// ", TOKEN).url == "https://x.ts.net/strata"
    assert ServerStore(path).get("A").url == "http://26.1.2.3:8095"


@pytest.mark.parametrize("name, url, token, field", [
    ("", "http://h:1", TOKEN, "name"),
    ("   ", "http://h:1", TOKEN, "name"),
    ("x" * 65, "http://h:1", TOKEN, "name"),
    ("Office", "http://h:1", TOKEN, "name"),          # a duplicate of the one saved first
    ("New", "ftp://x", TOKEN, "URL"),
    ("New", "http://", TOKEN, "URL"),
    ("New", "26.1.2.3:8095", TOKEN, "URL"),
    ("New", "http://h:99999", TOKEN, "URL"),
    ("New", "http://user:pw@h:1", TOKEN, "URL"),
    ("New", "http://h:1/?q=1", TOKEN, "URL"),
    ("New", "http://h:1", "two words", "token"),
    ("New", "http://h:1", "", "token"),
    ("New", "http://h:1", "ทดสอบ", "token"),
])
def test_bad_fields(path, name, url, token, field):
    s = ServerStore(path)
    s.save("Office", "http://26.1.2.3:8095", TOKEN)
    before = path.read_bytes()
    with pytest.raises(ValueError, match=field) as err:
        s.save(name, url, token)
    assert TOKEN not in str(err.value)
    assert not token or token not in str(err.value)               # an error never repeats a token
    assert path.read_bytes() == before
    assert [e.name for e in s.entries()] == ["Office"]


def test_name_64_is_fine(path):
    assert ServerStore(path).save("x" * 64, "http://h:1", TOKEN).name == "x" * 64


def test_save_without_token_keeps_it(path):
    s = ServerStore(path)
    s.save("Office", "http://26.1.2.3:8095", TOKEN)
    s.save("Office", "http://26.1.2.3:9000", None, old_name="Office")
    e = ServerStore(path).get("Office")
    assert e.url == "http://26.1.2.3:9000" and e.token.reveal() == TOKEN


def test_new_entry_without_token_is_missing(path):
    s = ServerStore(path)
    e = s.save("Office", "http://26.1.2.3:8095", None)
    assert e.token == Secret("")
    assert s.public() == [{"name": "Office", "url": "http://26.1.2.3:8095", "token": "missing"}]


def test_rename_keeps_token(path):
    s = ServerStore(path)
    s.save("Office", "http://26.1.2.3:8095", TOKEN)
    s.save("Second", "http://26.1.2.4:8095", "other-token")
    s.set_last("Office")
    s.save("Work", "http://26.1.2.3:8095", None, old_name="Office")
    r = ServerStore(path)
    assert [e.name for e in r.entries()] == ["Work", "Second"]      # keeps its place
    assert r.get("Work").token.reveal() == TOKEN and r.get("Office") is None
    assert r.last == "Work"


def test_rename_onto_another_name_refused(path):
    s = ServerStore(path)
    s.save("Office", "http://h:1", TOKEN)
    s.save("Second", "http://h:2", TOKEN)
    with pytest.raises(ValueError, match="name"):
        s.save("Second", "http://h:1", None, old_name="Office")


def test_rename_of_unknown_refused(path):
    with pytest.raises(ValueError, match="name"):
        ServerStore(path).save("New", "http://h:1", TOKEN, old_name="Nope")


def test_remove(path):
    s = ServerStore(path)
    s.save("Office", "http://h:1", TOKEN)
    s.set_last("Office")
    assert s.remove("Office") is True and s.remove("Office") is False
    r = ServerStore(path)
    assert r.entries() == [] and r.last is None


def test_set_last_unknown_refused(path):
    s = ServerStore(path)
    with pytest.raises(ValueError, match="name"):
        s.set_last("Nope")
    s.save("Office", "http://h:1", TOKEN)
    s.set_last("Office")
    s.set_last(None)
    assert ServerStore(path).last is None


def test_unreadable_blob_marked(path, monkeypatch):
    s = ServerStore(path)
    s.save("Office", "http://26.1.2.3:8095", TOKEN)
    s.save("Home", "http://h:1", None)

    def refuse(blob):
        raise dpapi.DpapiError("this blob is not for this Windows user")

    monkeypatch.setattr(dpapi, "unprotect", refuse)
    r = ServerStore(path)
    assert r.get("Office").token == Secret("")
    assert r.public() == [{"name": "Office", "url": "http://26.1.2.3:8095", "token": "unreadable"},
                          {"name": "Home", "url": "http://h:1", "token": "missing"}]


def test_unreadable_blob_kept_on_rewrite(path, monkeypatch):
    ServerStore(path).save("Office", "http://26.1.2.3:8095", TOKEN)
    blob = json.loads(path.read_text(encoding="utf-8"))["servers"][0]["token_dpapi"]
    real = dpapi.unprotect

    def refuse(b):
        raise dpapi.DpapiError("no")

    monkeypatch.setattr(dpapi, "unprotect", refuse)
    s = ServerStore(path)
    s.save("Other", "http://h:2", "t2")                  # rewrites the file
    s.save("Office", "http://26.1.2.3:9", None, old_name="Office")
    assert json.loads(path.read_text(encoding="utf-8"))["servers"][0]["token_dpapi"] == blob
    monkeypatch.setattr(dpapi, "unprotect", real)
    assert ServerStore(path).get("Office").token.reveal() == TOKEN


def test_public_has_no_token(path):
    s = ServerStore(path)
    s.save("Office", "http://26.1.2.3:8095", TOKEN)
    pub = s.public()
    assert pub == [{"name": "Office", "url": "http://26.1.2.3:8095", "token": "set"}]
    text = json.dumps(pub)
    assert TOKEN not in text and base64.b64encode(TOKEN.encode()).decode() not in text


def test_injected_round_trip(path):
    """P3-R3: a list of origins."""
    s = ServerStore(path)
    assert s.injected == []
    s.set_injected(["http://26.1.2.3:8095", "https://lab.example.ts.net"])
    assert ServerStore(path).injected == ["http://26.1.2.3:8095", "https://lab.example.ts.net"]
    got = s.injected
    got.append("x")                                              # a copy: the store is not changed through it
    assert s.injected == ["http://26.1.2.3:8095", "https://lab.example.ts.net"]
    s.set_injected([])
    assert ServerStore(path).injected == []


@pytest.mark.parametrize("old, read", [("http://26.1.2.3:8095", ["http://26.1.2.3:8095"]), (None, []), (7, []),
                                       (["a", 3, "", "a", "b"], ["a", "b"])])
def test_injected_older_forms(path, old, read):
    """The file before P3-R3 held one origin or null: read as a one-item list or none."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 1, "last": None, "injected": old, "servers": []}), encoding="utf-8")
    assert ServerStore(path).injected == read


def test_atomic_write(path, monkeypatch):
    s = ServerStore(path)
    s.save("Office", "http://26.1.2.3:8095", TOKEN)
    before = path.read_bytes()

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(servers.os, "replace", boom)
    with pytest.raises(OSError):
        s.save("Second", "http://h:2", TOKEN)
    assert path.read_bytes() == before
    assert [e.name for e in s.entries()] == ["Office"]           # memory follows the file
    assert not path.with_name(path.name + ".tmp").exists()


def test_missing_file_is_empty(path):
    s = ServerStore(path)
    assert s.entries() == [] and s.public() == [] and s.last is None and s.injected == []
    assert not path.exists()


def test_broken_file_kept_aside(path):
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    s = ServerStore(path)
    assert s.entries() == []
    assert path.with_name("servers.json.broken").read_text(encoding="utf-8") == "{not json"
