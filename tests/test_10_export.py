import numpy as np
import pickle
import tempfile
import os
import json
import pytest


def test_project_all_vectors():
    from scripts.export_10 import project_all_vectors

    sum_vectors = np.random.randn(100, 1536).astype(np.float32)
    coef = np.random.randn(300, 1536).astype(np.float32)
    intercept = np.random.randn(300).astype(np.float32)

    projected = project_all_vectors(sum_vectors, coef, intercept)

    assert projected.shape == (100, 300)
    assert projected.dtype == np.float16


def test_sumerian_lookup_find():
    from final_output.sumerian_lookup import SumerianLookup

    np.random.seed(42)
    n_sum = 5
    n_eng = 10
    dim = 300

    sum_vectors = np.random.randn(n_sum, dim).astype(np.float16)
    sum_vocab = ["lugal", "e2", "dingir", "an", "ki"]
    eng_vectors = np.random.randn(n_eng, dim).astype(np.float32)
    eng_vocab = [f"word_{i}" for i in range(n_eng)]

    with tempfile.TemporaryDirectory() as tmpdir:
        np.savez_compressed(
            os.path.join(tmpdir, "sumerian_aligned_vectors.npz"),
            vectors=sum_vectors,
        )
        with open(os.path.join(tmpdir, "sumerian_aligned_vocab.pkl"), "wb") as f:
            pickle.dump(sum_vocab, f)

        lookup = SumerianLookup(
            vectors_path=os.path.join(tmpdir, "sumerian_aligned_vectors.npz"),
            vocab_path=os.path.join(tmpdir, "sumerian_aligned_vocab.pkl"),
            glove_vectors=eng_vectors,
            glove_vocab=eng_vocab,
        )

        results = lookup.find("word_0", top_k=3)

        assert len(results) == 3
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
        assert results[0][0] in sum_vocab
