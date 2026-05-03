import json
import os
import tempfile

import numpy as np
import pytest


# --- SumerianLookup dual-view tests -----------------------------------------


def _build_tiny_lookup(tmpdir: str, seed: int = 42):
    """Build a SumerianLookup over tiny deliberately-aligned synthetic inputs.

    Sumerian vocab has 3 words. Each Sumerian word's aligned vector in the Gemma
    space deliberately matches English word_i in the Gemma space at index i, and
    similarly in the GloVe space -- so find("word_0", space=...) should return
    sum_0 at top-1 in both spaces.
    """
    from languages.sumerian.final_output.sumerian_lookup import SumerianLookup

    rng = np.random.default_rng(seed)

    n_sum = 3
    n_eng = 5
    gemma_dim = 768
    glove_dim = 300

    sum_vocab = ["sum_0", "sum_1", "sum_2"]

    eng_gemma = rng.standard_normal((n_eng, gemma_dim)).astype(np.float32)
    eng_vocab = [f"word_{i}" for i in range(n_eng)]

    sum_gemma = eng_gemma[:n_sum].astype(np.float16)

    eng_glove = rng.standard_normal((n_eng, glove_dim)).astype(np.float32)

    sum_glove = eng_glove[:n_sum].astype(np.float16)

    np.savez_compressed(
        os.path.join(tmpdir, "sumerian_aligned_gemma_vectors.npz"),
        vectors=sum_gemma,
    )
    np.savez_compressed(
        os.path.join(tmpdir, "sumerian_aligned_vectors.npz"),
        vectors=sum_glove,
    )
    with open(os.path.join(tmpdir, "sumerian_aligned_vocab.pkl"), "wb") as f:
        import pickle as _pkl
        _pkl.dump(sum_vocab, f)
    np.savez_compressed(
        os.path.join(tmpdir, "english_gemma_whitened_768d.npz"),
        vocab=np.array(eng_vocab),
        vectors=eng_gemma,
    )

    return SumerianLookup(
        gemma_vectors_path=os.path.join(tmpdir, "sumerian_aligned_gemma_vectors.npz"),
        glove_vectors_path=os.path.join(tmpdir, "sumerian_aligned_vectors.npz"),
        vocab_path=os.path.join(tmpdir, "sumerian_aligned_vocab.pkl"),
        gemma_english_path=os.path.join(tmpdir, "english_gemma_whitened_768d.npz"),
        glove_english_vectors=eng_glove,
        glove_english_vocab=eng_vocab,
    ), sum_vocab, eng_vocab


def test_sumerian_lookup_find_gemma_returns_top_k():
    with tempfile.TemporaryDirectory() as tmpdir:
        lookup, sum_vocab, eng_vocab = _build_tiny_lookup(tmpdir)
        results = lookup.find("word_0", top_k=3, space="gemma")
        assert len(results) == 3
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
        assert results[0][0] == "sum_0"
        assert results[0][1] > 0.99


def test_sumerian_lookup_find_glove_returns_top_k():
    with tempfile.TemporaryDirectory() as tmpdir:
        lookup, sum_vocab, eng_vocab = _build_tiny_lookup(tmpdir)
        results = lookup.find("word_1", top_k=3, space="glove")
        assert len(results) == 3
        assert results[0][0] == "sum_1"
        assert results[0][1] > 0.99


def test_sumerian_lookup_find_both_has_both_keys():
    with tempfile.TemporaryDirectory() as tmpdir:
        lookup, _, _ = _build_tiny_lookup(tmpdir)
        result = lookup.find_both("word_2", top_k=2)
        assert set(result.keys()) == {"gemma", "glove"}
        assert len(result["gemma"]) == 2
        assert len(result["glove"]) == 2
        assert result["gemma"][0][0] == "sum_2"
        assert result["glove"][0][0] == "sum_2"


def test_sumerian_lookup_unknown_space_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        lookup, _, _ = _build_tiny_lookup(tmpdir)
        with pytest.raises(ValueError, match="space must be"):
            lookup.find("word_0", space="bert")


def test_sumerian_lookup_oov_returns_empty_list():
    with tempfile.TemporaryDirectory() as tmpdir:
        lookup, _, _ = _build_tiny_lookup(tmpdir)
        assert lookup.find("definitely_not_a_word_xyz", space="gemma") == []
        assert lookup.find("definitely_not_a_word_xyz", space="glove") == []


def test_sumerian_lookup_analogy_routes_by_space():
    with tempfile.TemporaryDirectory() as tmpdir:
        lookup, _, _ = _build_tiny_lookup(tmpdir)
        gemma_result = lookup.find_analogy("word_0", "word_1", "word_2", top_k=3, space="gemma")
        glove_result = lookup.find_analogy("word_0", "word_1", "word_2", top_k=3, space="glove")
        assert len(gemma_result) > 0
        assert len(glove_result) > 0
        assert [w for w, _ in gemma_result] != [w for w, _ in glove_result]


def test_sumerian_lookup_blend_empty_weights_returns_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        lookup, _, _ = _build_tiny_lookup(tmpdir)
        assert lookup.find_blend({"unknown_word_1": 1.0}, space="gemma") == []
        assert lookup.find_blend({}, space="gemma") == []


# --- Export tests (covered in Task 2) --------------------------------------


def test_project_all_vectors():
    from languages.sumerian.scripts.export_10 import project_all_vectors

    sum_vectors = np.random.randn(100, 1536).astype(np.float32)
    coef = np.random.randn(300, 1536).astype(np.float32)
    intercept = np.random.randn(300).astype(np.float32)

    projected = project_all_vectors(sum_vectors, coef, intercept)

    assert projected.shape == (100, 300)
    assert projected.dtype == np.float16


def test_export_writes_both_spaces_and_v2_metadata(tmp_path, monkeypatch):
    """End-to-end: export script produces both aligned npz files and v2 metadata."""
    import languages.sumerian.scripts.export_10 as export_10_module

    n_sum = 4
    fused_dim = 1536
    glove_dim = 300
    gemma_dim = 768

    rng = np.random.default_rng(7)
    sum_vocab = ["a", "b", "c", "d"]
    fused = rng.standard_normal((n_sum, fused_dim)).astype(np.float32)

    glove_coef = rng.standard_normal((glove_dim, fused_dim)).astype(np.float32)
    glove_intercept = rng.standard_normal(glove_dim).astype(np.float32)

    gemma_coef = rng.standard_normal((gemma_dim, fused_dim)).astype(np.float32)
    gemma_intercept = rng.standard_normal(gemma_dim).astype(np.float32)

    models = tmp_path / "models"
    results = tmp_path / "results"
    final = tmp_path / "final_output"
    models.mkdir()
    results.mkdir()

    np.savez_compressed(
        str(models / "fused_embeddings_1536d.npz"),
        vectors=fused,
        vocab=np.array(sum_vocab),
    )
    np.savez_compressed(
        str(models / "ridge_weights.npz"),
        coef=glove_coef,
        intercept=glove_intercept,
    )
    np.savez_compressed(
        str(models / "ridge_weights_gemma_whitened.npz"),
        coef=gemma_coef,
        intercept=gemma_intercept,
    )
    (results / "alignment_results.json").write_text(json.dumps({
        "accuracy": {"top1": 17.30, "top5": 22.90, "top10": 25.19},
        "config": {
            "alignment": "Ridge", "alpha": 100, "train_size": 1572,
            "test_size": 393, "valid_anchors": 1965, "total_anchors": 13886,
            "sumerian_vocab": 35508, "fused_dim": 1536,
        },
    }))
    (results / "alignment_results_gemma_whitened.json").write_text(json.dumps({
        "accuracy": {"top1": 19.85, "top5": 23.66, "top10": 26.21},
        "config": {
            "alignment": "Ridge", "alpha": 100, "mode": "whitened",
            "gemma_model": "google/embeddinggemma-300m", "gloss_hit_rate": 21.39,
            "test_size_count": 393, "train_size": 1572, "valid_anchors": 1965,
            "total_anchors": 13886, "random_state": 42,
        },
    }))

    monkeypatch.setattr(export_10_module, "MODELS_DIR", models)
    monkeypatch.setattr(export_10_module, "RESULTS_DIR", results)
    monkeypatch.setattr(export_10_module, "FINAL_OUTPUT", final)
    export_10_module.main()

    assert (final / "sumerian_aligned_vectors.npz").exists()
    assert (final / "sumerian_aligned_gemma_vectors.npz").exists()
    assert (final / "sumerian_aligned_vocab.pkl").exists()
    assert (final / "metadata.json").exists()

    glove_npz = np.load(str(final / "sumerian_aligned_vectors.npz"))
    gemma_npz = np.load(str(final / "sumerian_aligned_gemma_vectors.npz"))
    assert glove_npz["vectors"].shape == (n_sum, glove_dim)
    assert gemma_npz["vectors"].shape == (n_sum, gemma_dim)
    assert glove_npz["vectors"].dtype == np.float16
    assert gemma_npz["vectors"].dtype == np.float16

    metadata = json.loads((final / "metadata.json").read_text())
    assert metadata["schema_version"] == 2
    assert metadata["shared"]["vocab_size"] == n_sum
    assert metadata["spaces"]["gemma"]["dim"] == 768
    assert metadata["spaces"]["glove"]["dim"] == 300
    assert metadata["spaces"]["gemma"]["ridge_alpha"] == 100
    assert metadata["spaces"]["glove"]["ridge_alpha"] == 100
    assert metadata["spaces"]["gemma"]["accuracy"]["top1"] == 19.85
    assert metadata["spaces"]["glove"]["accuracy"]["top1"] == 17.30
    assert metadata["shared"]["test_size_count"] == 393
    assert metadata["shared"]["train_size"] == 1572
