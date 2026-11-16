"""desktop/paths.py: where Strata Desktop keeps its files (#310 part 3 task 1)."""
import os
from pathlib import Path

from desktop import paths


def test_override(tmp_path):
    assert paths.data_dir(str(tmp_path / "d")) == tmp_path / "d"


def test_default(monkeypatch, tmp_path):
    if os.name == "nt":
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
        assert paths.data_dir() == tmp_path / "StrataDesktop"
    else:
        assert paths.data_dir() == Path.home() / ".strata-desktop"


def test_files_inside(tmp_path):
    assert paths.servers_file(tmp_path) == tmp_path / "servers.json"
    assert paths.webview_dir(tmp_path) == tmp_path / "webview"
    assert paths.log_file(tmp_path) == tmp_path / "desktop.log"


def test_runner_dir(monkeypatch, tmp_path):
    from serve import runner
    if os.name == "nt":
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    assert paths.runner_dir(tmp_path, override_used=False) == runner.runner_dir()
    assert paths.runner_dir(tmp_path, override_used=True) == tmp_path / "runner"


def test_paths_create_nothing(tmp_path):
    paths.data_dir(str(tmp_path / "d"))
    paths.runner_dir(tmp_path / "d", override_used=True)
    assert list(tmp_path.iterdir()) == []
