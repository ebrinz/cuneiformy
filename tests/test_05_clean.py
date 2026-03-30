import pytest


def test_clean_atf_line():
    """Clean ATF editorial markers and normalize."""
    from scripts.clean_05 import clean_atf_line

    assert clean_atf_line("1(disz) geme2 u4 1(disz)-sze3") == "1(disz) geme2 u4 1(disz) sze3"
    assert clean_atf_line("ki-masz{ki}") == "ki masz ki"
    assert clean_atf_line("[lugal]-e") == "lugal e"
    assert clean_atf_line("mu!(BU)") == "mu"
    assert clean_atf_line("dingir#") == "dingir"
    assert clean_atf_line("a?-ba") == "a ba"


def test_clean_atf_line_whitespace():
    """Multiple spaces should collapse to one."""
    from scripts.clean_05 import clean_atf_line

    assert clean_atf_line("lugal    e2") == "lugal e2"


def test_clean_atf_line_empty():
    """Lines that become empty after cleaning should return empty string."""
    from scripts.clean_05 import clean_atf_line

    assert clean_atf_line("[...]") == ""
    assert clean_atf_line("$ broken") == ""


def test_build_corpus():
    """Build cleaned corpus from merged texts."""
    from scripts.clean_05 import build_corpus

    texts = [
        {"lines": ["lugal-e mu-un-na-ni-ib-gi4-gi4", "e2-gal-la ba-an-ku4"]},
        {"lines": ["dingir gal-gal-e-ne"]},
    ]

    lines = build_corpus(texts)

    assert len(lines) == 2
    assert "lugal" in lines[0]
