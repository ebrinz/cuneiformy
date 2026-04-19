import pytest


def test_subscripts_to_ascii():
    from scripts.sumerian_normalize import normalize_sumerian_token
    assert normalize_sumerian_token("hulum₂") == "hulum2"
    assert normalize_sumerian_token("₀₁₂₃₄₅₆₇₈₉") == "0123456789"


def test_strips_determinative_braces():
    from scripts.sumerian_normalize import normalize_sumerian_token
    assert normalize_sumerian_token("{tug₂}mug") == "tug2mug"
    # Multiple braces in one token collapse correctly.
    assert normalize_sumerian_token("{d}{ki}enlil") == "dkienlil"


def test_oracc_to_atf_letters():
    from scripts.sumerian_normalize import normalize_sumerian_token
    # All lowercase variants, with the aleph U+02BE dropped to empty.
    assert normalize_sumerian_token("šeš") == "szesz"
    assert normalize_sumerian_token("ḫar") == "har"
    assert normalize_sumerian_token("ṣabu") == "sabu"
    assert normalize_sumerian_token("ṭub") == "tub"
    assert normalize_sumerian_token("ŋar") == "jar"
    assert normalize_sumerian_token("ʾa3") == "a3"
    # Uppercase variants lowercase-normalize after letter map.
    assert normalize_sumerian_token("Š") == "sz"
    assert normalize_sumerian_token("Ḫ") == "h"


def test_drops_hyphens():
    from scripts.sumerian_normalize import normalize_sumerian_token
    assert normalize_sumerian_token("nar-ta") == "narta"
    assert normalize_sumerian_token("za₃-sze₃-la₂") == "za3sze3la2"
    assert normalize_sumerian_token("mu-du₃-sze₃") == "mudu3sze3"


def test_lowercases():
    from scripts.sumerian_normalize import normalize_sumerian_token
    assert normalize_sumerian_token("LUGAL") == "lugal"
    assert normalize_sumerian_token("Dingir") == "dingir"


def test_strips_whitespace():
    from scripts.sumerian_normalize import normalize_sumerian_token
    assert normalize_sumerian_token(" lugal ") == "lugal"
    assert normalize_sumerian_token("\tdingir\n") == "dingir"


def test_handles_empty_and_none():
    from scripts.sumerian_normalize import normalize_sumerian_token
    assert normalize_sumerian_token("") == ""
    assert normalize_sumerian_token(None) == ""


def test_idempotent():
    from scripts.sumerian_normalize import normalize_sumerian_token
    for raw in ("lugal", "{tug₂}mug", "za₃-sze₃-la₂", "ŠEŠ", "ʾan-na"):
        once = normalize_sumerian_token(raw)
        twice = normalize_sumerian_token(once)
        assert once == twice, f"not idempotent on {raw!r}: {once!r} -> {twice!r}"


def test_combined_chain():
    from scripts.sumerian_normalize import normalize_sumerian_token
    # Subscripts + braces + ORACC letters + hyphens + uppercase, all at once.
    assert normalize_sumerian_token("{Tug₂}-Sze₃-la₂") == "tug2sze3la2"
    assert normalize_sumerian_token("{D}Šeš₂-Ŋar") == "dszesz2jar"
