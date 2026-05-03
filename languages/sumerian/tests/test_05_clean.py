import pytest


def test_clean_atf_line():
    """Clean ATF editorial markers and normalize transliteration."""
    from languages.sumerian.scripts.clean_05 import clean_atf_line

    # Editorial cleanup (corpus stays in ATF convention)
    # Hyphens emit both compound + split: "ki-masz" -> "kimasz ki masz"
    result = clean_atf_line("1(disz) geme2 u4 1(disz)-sze3")
    assert "geme2" in result
    assert "sze3" in result
    result = clean_atf_line("ki-masz{ki}")
    assert "ki" in result
    assert "masz" in result
    result = clean_atf_line("[lugal]-e")
    assert "lugal" in result
    assert "e" in result.split()
    assert clean_atf_line("mu!(BU)") == "mu"
    assert clean_atf_line("dingir#") == "dingir"
    result = clean_atf_line("a?-ba")
    assert "a" in result.split()
    assert "ba" in result.split()


def test_clean_atf_removes_compound_signs():
    """Compound signs and semantic markers should be removed."""
    from languages.sumerian.scripts.clean_05 import clean_atf_line

    assert "|" not in clean_atf_line("|GA2xAN| lugal")
    assert "_" not in clean_atf_line("_d_nin lugal")
    assert "<" not in clean_atf_line("<szum2> lugal")


def test_clean_atf_normalizes_transliteration():
    """Normalization should lowercase and convert subscript digits, but preserve ATF convention."""
    from languages.sumerian.scripts.clean_05 import normalize_transliteration

    assert normalize_transliteration("sze") == "sze"
    assert normalize_transliteration("szu") == "szu"
    assert normalize_transliteration("lu₂") == "lu2"
    assert normalize_transliteration("e₂") == "e2"


def test_clean_atf_line_whitespace():
    """Multiple spaces should collapse to one."""
    from languages.sumerian.scripts.clean_05 import clean_atf_line

    assert clean_atf_line("lugal    e2") == "lugal e2"


def test_clean_atf_line_empty():
    """Lines that become empty after cleaning should return empty string."""
    from languages.sumerian.scripts.clean_05 import clean_atf_line

    assert clean_atf_line("[...]") == ""
    assert clean_atf_line("$ broken") == ""


def test_build_corpus():
    """Build cleaned corpus from merged texts."""
    from languages.sumerian.scripts.clean_05 import build_corpus

    texts = [
        {"lines": ["lugal-e mu-un-na-ni-ib-gi4-gi4", "e2-gal-la ba-an-ku4"]},
        {"lines": ["dingir gal-gal-e-ne"]},
    ]

    lines = build_corpus(texts)

    assert len(lines) == 2
    assert "lugal" in lines[0]
