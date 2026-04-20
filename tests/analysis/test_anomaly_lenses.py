"""Unit tests for the civilization-agnostic anomaly atlas framework + lenses."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


def test_anomaly_config_is_frozen():
    from scripts.analysis.anomaly_framework import AnomalyConfig

    config = AnomalyConfig(
        civilization_name="test",
        aligned_gemma_path=Path("/tmp/g.npz"),
        aligned_glove_path=None,
        source_vocab_path=Path("/tmp/vocab.pkl"),
        target_gemma_vocab_path=Path("/tmp/egm.npz"),
        target_glove_vocab_path=None,
        anchors_path=Path("/tmp/a.json"),
        corpus_frequency_path=Path("/tmp/corp.txt"),
        junk_target_glosses=frozenset({"x", "n"}),
        min_anchor_confidence=0.5,
        min_token_length=2,
        output_atlas_json=Path("/tmp/out.json"),
        output_markdown_dir=Path("/tmp/md"),
        output_figures_dir=None,
    )
    with pytest.raises((AttributeError, Exception)):
        config.civilization_name = "hacked"


def _normalize_rows(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


# --- Lens 1: English displacement ---------------------------------------


def test_lens1_ranks_low_cosine_first():
    from scripts.analysis.anomaly_lenses import lens1_english_displacement

    # Two anchors: anchor[0] has identical source/target dirs (cos=1),
    # anchor[1] has orthogonal (cos=0). Top row should be anchor[1].
    aligned = _normalize_rows(np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32))
    source_vocab = ["a_src", "b_src"]
    target_vectors = _normalize_rows(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    target_vocab_map = {"a_tgt": 0, "b_tgt": 1}
    anchors = [
        {"sumerian": "a_src", "english": "a_tgt", "confidence": 0.9, "source": "ePSD2"},
        {"sumerian": "b_src", "english": "b_tgt", "confidence": 0.9, "source": "ePSD2"},
    ]
    result = lens1_english_displacement(
        aligned, source_vocab, target_vectors, target_vocab_map, anchors,
        top_n=10, junk_target_glosses=frozenset(), min_token_length=2,
        min_anchor_confidence=0.5,
    )
    rows = result["rows_unfiltered"]
    assert len(rows) == 2
    assert rows[0]["sumerian"] == "b_src"
    assert rows[0]["cosine_similarity"] == pytest.approx(0.0, abs=1e-5)


def test_lens1_filtered_excludes_short_english():
    from scripts.analysis.anomaly_lenses import lens1_english_displacement

    aligned = _normalize_rows(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    source_vocab = ["alpha", "beta"]
    target_vectors = _normalize_rows(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    target_vocab_map = {"king": 0, "c": 1}
    anchors = [
        {"sumerian": "alpha", "english": "king", "confidence": 0.9, "source": "ePSD2"},
        {"sumerian": "beta",  "english": "c",    "confidence": 0.9, "source": "ePSD2"},
    ]
    result = lens1_english_displacement(
        aligned, source_vocab, target_vectors, target_vocab_map, anchors,
        top_n=10, junk_target_glosses=frozenset(), min_token_length=2,
        min_anchor_confidence=0.5,
    )
    # Filtered tier drops the english="c" row.
    filtered_english = {r["english"] for r in result["rows_filtered"]}
    assert "c" not in filtered_english
    unfiltered_english = {r["english"] for r in result["rows_unfiltered"]}
    assert "c" in unfiltered_english


# --- Lens 3: Isolation --------------------------------------------------


def test_lens3_isolation_is_k_nearest_distance():
    from scripts.analysis.anomaly_lenses import lens3_isolation

    # 5 tokens in 2D: four form a tight cluster, one is isolated.
    vectors = np.array([
        [1.0, 0.0],
        [0.99, 0.05],
        [0.98, -0.05],
        [1.0, 0.1],
        [-1.0, 0.0],   # isolated — opposite direction
    ], dtype=np.float32)
    aligned = _normalize_rows(vectors)
    vocab = ["a", "b", "c", "d", "iso"]
    result = lens3_isolation(aligned, vocab, isolation_k=1, top_n=5)
    rows = result["rows"]
    assert rows[0]["sumerian"] == "iso"
    # cosine distance between iso and the cluster tokens is > 1.5
    assert rows[0]["distance_to_kth_neighbor"] > 1.0


def test_lens3_returns_histogram():
    from scripts.analysis.anomaly_lenses import lens3_isolation

    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((20, 8)).astype(np.float32)
    aligned = _normalize_rows(vectors)
    vocab = [f"t{i}" for i in range(20)]
    result = lens3_isolation(aligned, vocab, isolation_k=3, top_n=10)
    assert "histogram" in result
    assert "bin_edges" in result["histogram"]
    assert "counts" in result["histogram"]
    assert sum(result["histogram"]["counts"]) == len(aligned)
