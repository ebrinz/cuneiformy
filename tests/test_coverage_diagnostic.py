import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


# --- Loader tests ----------------------------------------------------------


def test_load_corpus_frequency_counts_tokens():
    from scripts.coverage_diagnostic import _load_corpus_frequency

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cleaned.txt"
        path.write_text("lugal dingir lugal\nlugal mu-dug\ndingir\n", encoding="utf-8")
        freq = _load_corpus_frequency(path)
        assert freq["lugal"] == 3
        assert freq["dingir"] == 2
        assert freq["mu-dug"] == 1


def test_load_corpus_frequency_empty_file():
    from scripts.coverage_diagnostic import _load_corpus_frequency

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "empty.txt"
        path.write_text("", encoding="utf-8")
        freq = _load_corpus_frequency(path)
        assert len(freq) == 0


def test_load_lemma_surface_map_builds_citation_to_surfaces():
    from scripts.coverage_diagnostic import _load_lemma_surface_map

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "lemmas.json"
        lemmas = [
            {"cf": "lugal", "form": "lugal", "gw": "king"},
            {"cf": "lugal", "form": "lugale", "gw": "king"},
            {"cf": "lugal", "form": "lugalanene", "gw": "king"},
            {"cf": "dingir", "form": "dingir", "gw": "god"},
        ]
        path.write_text(json.dumps(lemmas), encoding="utf-8")
        mapping = _load_lemma_surface_map(path)
        assert mapping["lugal"] == {"lugal", "lugale", "lugalanene"}
        assert mapping["dingir"] == {"dingir"}


def test_load_lemma_surface_map_normalizes_via_oracc_to_atf():
    from scripts.coverage_diagnostic import _load_lemma_surface_map

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "lemmas.json"
        lemmas = [
            {"cf": "šeš", "form": "šeš", "gw": "brother"},
            {"cf": "šeš", "form": "šešeš", "gw": "brother"},
        ]
        path.write_text(json.dumps(lemmas, ensure_ascii=False), encoding="utf-8")
        mapping = _load_lemma_surface_map(path)
        # ORACC 'š' -> ATF 'sz' normalization applied
        assert "szesz" in mapping
        assert "szesz" in mapping["szesz"]
        assert "szeszesz" in mapping["szesz"]


def test_load_lemma_surface_map_skips_empty_cf_or_form():
    from scripts.coverage_diagnostic import _load_lemma_surface_map

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "lemmas.json"
        lemmas = [
            {"cf": "lugal", "form": "lugal", "gw": "king"},
            {"cf": "", "form": "lugal", "gw": "king"},
            {"cf": "lugal", "form": "", "gw": "king"},
        ]
        path.write_text(json.dumps(lemmas), encoding="utf-8")
        mapping = _load_lemma_surface_map(path)
        assert mapping == {"lugal": {"lugal"}}


def test_diagnostic_context_is_frozen():
    from scripts.coverage_diagnostic import DiagnosticContext

    ctx = _make_tiny_context()
    with pytest.raises((AttributeError, Exception)):
        ctx.fused_vocab = frozenset()


def _make_tiny_context():
    """Build a DiagnosticContext over tiny synthetic inputs for downstream tests."""
    from scripts.coverage_diagnostic import DiagnosticContext

    return DiagnosticContext(
        fused_vocab=frozenset({"lugal", "dingir", "nar", "ta", "narta"}),
        glove_vocab=frozenset({"king", "god"}),
        gemma_vocab=frozenset({"king", "god"}),
        corpus_frequency={"lugal": 10, "dingir": 8, "narta": 2, "mu": 1},
        lemma_surface_map={"lugal": frozenset({"lugal", "lugale"})},
        fasttext_model=None,
        gemma_english_vocab=["king", "god"],
        gemma_english_vectors=np.eye(2, 768, dtype=np.float32),
        ridge_gemma_coef=np.zeros((768, 1536), dtype=np.float32),
        ridge_gemma_intercept=np.zeros(768, dtype=np.float32),
    )


def test_diagnostic_context_accepts_all_required_fields():
    ctx = _make_tiny_context()
    assert "lugal" in ctx.fused_vocab
    assert ctx.corpus_frequency["dingir"] == 8
    assert "lugal" in ctx.lemma_surface_map
    assert ctx.ridge_gemma_coef.shape == (768, 1536)
