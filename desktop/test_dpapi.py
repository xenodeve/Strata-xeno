"""desktop/dpapi.py: the token store's encryption with Windows DPAPI (#310 part 3 task 1)."""
import os

import pytest

from desktop import dpapi

pytestmark = pytest.mark.skipif(os.name != "nt", reason="DPAPI is Windows only")


def test_round_trip():
    data = b"\x00ab\x00c" + "ทดสอบ é".encode("utf-8") + bytes(range(256))
    blob = dpapi.protect(data)
    assert blob != data
    assert dpapi.unprotect(blob) == data


def test_empty_round_trip():
    assert dpapi.unprotect(dpapi.protect(b"")) == b""


def test_tampered_blob_raises():
    blob = bytearray(dpapi.protect(b"sk-strata-token-123"))
    blob[-5] ^= 0x01                      # inside the HMAC at the end: the blob no longer verifies
    with pytest.raises(dpapi.DpapiError):
        dpapi.unprotect(bytes(blob))


def test_garbage_raises():
    with pytest.raises(dpapi.DpapiError):
        dpapi.unprotect(b"not a dpapi blob")


def test_blob_is_not_plaintext():
    token = b"sk-plain-token-ABCDEFGHIJKLMNOP"
    assert token not in dpapi.protect(token)
