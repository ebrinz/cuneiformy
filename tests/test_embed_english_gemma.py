import pytest


def test_format_gloss_with_definition():
    from scripts.embed_english_gemma import format_gloss

    result = format_gloss("flour", "a fine powder made by grinding grain")
    assert result == "flour: a fine powder made by grinding grain"


def test_format_gloss_bare_when_no_definition():
    from scripts.embed_english_gemma import format_gloss

    result = format_gloss("flour", None)
    assert result == "flour"


def test_format_gloss_bare_when_empty_definition():
    from scripts.embed_english_gemma import format_gloss

    result = format_gloss("flour", "")
    assert result == "flour"


def test_lookup_gloss_known_hit():
    from scripts.embed_english_gemma import lookup_gloss

    definition = lookup_gloss("king")
    assert definition is not None
    assert isinstance(definition, str)
    assert len(definition) > 0


def test_lookup_gloss_known_miss():
    from scripts.embed_english_gemma import lookup_gloss

    definition = lookup_gloss("zzzzqqqqnotaword12345")
    assert definition is None
