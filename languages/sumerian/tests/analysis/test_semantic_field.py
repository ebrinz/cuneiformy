import tempfile
from pathlib import Path

import numpy as np
import pytest


def _synthetic_lookup(tokens, vectors):
    """Minimal SumerianLookup stub exposing the _spaces dict our functions use."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = vectors / norms

    class Stub:
        def __init__(self):
            self.vocab = tokens
            self._spaces = {
                "gemma": {"sum_norm": normalized, "sum_dim": vectors.shape[1]},
            }
            self._token_to_idx = {t: i for i, t in enumerate(tokens)}
    return Stub()


def test_pairwise_distances_is_symmetric():
    from framework.analysis.semantic_field import compute_pairwise_distances

    tokens = ["a", "b", "c"]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    lookup = _synthetic_lookup(tokens, vectors)

    D = compute_pairwise_distances(lookup, tokens, space="gemma")
    assert np.allclose(D, D.T)


def test_pairwise_distances_diagonal_is_zero():
    from framework.analysis.semantic_field import compute_pairwise_distances

    tokens = ["a", "b"]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    lookup = _synthetic_lookup(tokens, vectors)

    D = compute_pairwise_distances(lookup, tokens, space="gemma")
    assert np.allclose(np.diag(D), 0.0)


def test_pairwise_distances_shape_matches_input():
    from framework.analysis.semantic_field import compute_pairwise_distances

    tokens = ["a", "b", "c", "d"]
    vectors = np.random.default_rng(0).standard_normal((4, 8)).astype(np.float32)
    lookup = _synthetic_lookup(tokens, vectors)

    D = compute_pairwise_distances(lookup, tokens, space="gemma")
    assert D.shape == (4, 4)


def test_pairwise_distances_raises_on_unknown_token():
    from framework.analysis.semantic_field import compute_pairwise_distances

    tokens = ["a", "b"]
    vectors = np.eye(2, dtype=np.float32)
    lookup = _synthetic_lookup(tokens, vectors)

    with pytest.raises(KeyError, match="unknown"):
        compute_pairwise_distances(lookup, ["a", "unknown"], space="gemma")


def test_heatmap_writes_png():
    from framework.analysis.semantic_field import render_semantic_field_heatmap

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "heatmap.png"
        distances = np.array([[0.0, 0.3, 0.5], [0.3, 0.0, 0.2], [0.5, 0.2, 0.0]])
        render_semantic_field_heatmap(
            distances, ["a", "b", "c"], title="test", out_path=out_path
        )
        assert out_path.exists()
        assert out_path.stat().st_size > 0
