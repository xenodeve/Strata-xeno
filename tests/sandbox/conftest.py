"""The folders the tests make under the scratch root go when the tests end. A denied one keeps its ACL for good on purpose,
so nothing else would remove it."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@pytest.fixture(scope="session", autouse=True)
def clean_shared_folders():
    if sys.platform != "win32":
        yield
        return
    from serve.sandbox.isolation import Scratch, _remove_tree
    shared = Scratch().root / "shared"
    before = {entry.name for entry in shared.iterdir()}
    yield
    for entry in shared.iterdir():
        if entry.name not in before:
            _remove_tree(entry)
