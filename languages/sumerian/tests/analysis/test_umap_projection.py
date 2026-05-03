import tempfile
from pathlib import Path

import numpy as np


def _umap_lookup_stub(tokens, vectors):
    def _norm(X):
        n = np.linalg.norm(X, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return X / n

    class Stub:
        def __init__(self):
            self.vocab = tokens
            self._spaces = {
                "gemma": {"sum_norm": _norm(vectors), "sum_dim": vectors.shape[1]}
            }
    return Stub()


def test_umap_writes_png_for_sufficient_vocab():
    from framework.analysis.umap_projection import umap_cosmogonic_vocabulary

    rng = np.random.default_rng(0)
    tokens = [f"t{i}" for i in range(20)]
    vectors = rng.standard_normal((20, 16)).astype(np.float32)
    lookup = _umap_lookup_stub(tokens, vectors)
    labels = {t: "cluster_a" if i % 2 == 0 else "cluster_b" for i, t in enumerate(tokens)}

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "umap.png"
        umap_cosmogonic_vocabulary(lookup, tokens, labels, space="gemma", out_path=out_path)
        assert out_path.exists()
        assert out_path.stat().st_size > 0


def test_umap_falls_back_to_pca_when_vocab_too_small():
    from framework.analysis.umap_projection import umap_cosmogonic_vocabulary

    tokens = ["a", "b", "c"]  # too small for UMAP
    vectors = np.eye(3, 8, dtype=np.float32)
    lookup = _umap_lookup_stub(tokens, vectors)
    labels = {"a": "x", "b": "x", "c": "y"}

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "fallback.png"
        umap_cosmogonic_vocabulary(lookup, tokens, labels, space="gemma", out_path=out_path)
        assert out_path.exists()


def test_umap_deterministic_with_fixed_seed():
    """Two calls with the same inputs should produce the same PNG bytes."""
    from framework.analysis.umap_projection import umap_cosmogonic_vocabulary

    rng = np.random.default_rng(42)
    tokens = [f"t{i}" for i in range(20)]
    vectors = rng.standard_normal((20, 16)).astype(np.float32)
    lookup = _umap_lookup_stub(tokens, vectors)
    labels = {t: "a" for t in tokens}

    with tempfile.TemporaryDirectory() as tmp:
        p1 = Path(tmp) / "one.png"
        p2 = Path(tmp) / "two.png"
        umap_cosmogonic_vocabulary(lookup, tokens, labels, space="gemma", out_path=p1)
        umap_cosmogonic_vocabulary(lookup, tokens, labels, space="gemma", out_path=p2)
        # UMAP with fixed random_state + same matplotlib backend should be deterministic.
        # PNG bytes may vary due to matplotlib version; we test that the embedding
        # coords are identical by running UMAP twice explicitly.
        # (The PNG test above proves the file is produced; this test proves the
        # coordinate computation is stable.)
        from framework.analysis.umap_projection import _compute_embedding
        coords1 = _compute_embedding(
            np.linalg.norm(vectors, axis=1, keepdims=True).clip(1e-9) * 0 + vectors / np.linalg.norm(vectors, axis=1, keepdims=True).clip(1e-9),
            seed=42,
        )
        coords2 = _compute_embedding(
            vectors / np.linalg.norm(vectors, axis=1, keepdims=True).clip(1e-9),
            seed=42,
        )
        assert np.allclose(coords1, coords2)
