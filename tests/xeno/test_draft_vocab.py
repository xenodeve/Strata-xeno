"""#55 W7: the MTP draft head proposes only tokens in rt/draft_vocab.bin.  The shipped subset held 14 of the
vocabulary's 5,741 Thai tokens, so a Thai answer drafted almost nothing (the Thai twin of upstream's #137); with them
added, Thai decode measured +20-35 % with the same output.  tools/draft_vocab.py must know the Thai script."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
import draft_vocab as dv  # noqa: E402


def test_thai_text_is_the_thai_script():
    assert dv.scripts_of("สวัสดีครับ") == {"thai"}


def test_thai_is_not_mistaken_for_cjk_or_latin():
    assert dv.scripts_of("hello") == set()
    assert "thai" not in dv.scripts_of("你好")


def test_the_thai_group_can_be_added():
    assert "thai" in dv.SCRIPTS
