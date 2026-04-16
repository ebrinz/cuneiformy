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


def test_load_glove_vocab_reads_words_only(tmp_path):
    from scripts.embed_english_gemma import load_glove_vocab

    glove_file = tmp_path / "fake_glove.txt"
    glove_file.write_text(
        "king 0.1 0.2 0.3\n"
        "queen 0.4 0.5 0.6\n"
        "flour 0.7 0.8 0.9\n"
    )

    vocab = load_glove_vocab(glove_file)
    assert vocab == ["king", "queen", "flour"]


def test_output_is_up_to_date_missing_file(tmp_path):
    from scripts.embed_english_gemma import output_is_up_to_date

    missing = tmp_path / "nope.npz"
    assert output_is_up_to_date(missing, ["a", "b"]) is False


def test_output_is_up_to_date_match(tmp_path):
    import numpy as np
    from scripts.embed_english_gemma import output_is_up_to_date

    path = tmp_path / "cache.npz"
    np.savez_compressed(
        str(path),
        vocab=np.array(["a", "b", "c"]),
        vectors=np.zeros((3, 768), dtype=np.float32),
    )
    assert output_is_up_to_date(path, ["a", "b", "c"]) is True


def test_output_is_up_to_date_vocab_mismatch(tmp_path):
    import numpy as np
    from scripts.embed_english_gemma import output_is_up_to_date

    path = tmp_path / "cache.npz"
    np.savez_compressed(
        str(path),
        vocab=np.array(["a", "b", "c"]),
        vectors=np.zeros((3, 768), dtype=np.float32),
    )
    assert output_is_up_to_date(path, ["a", "b", "different"]) is False
