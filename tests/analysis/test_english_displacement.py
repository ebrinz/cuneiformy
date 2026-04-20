import numpy as np
import pytest


def _displacement_lookup_stub(eng_vocab, eng_vectors, sum_vocab, sum_vectors_gemma):
    """SumerianLookup stub exposing _spaces with eng_norm and sum_norm."""
    def _norm(X):
        n = np.linalg.norm(X, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return X / n

    class Stub:
        def __init__(self):
            self.vocab = sum_vocab
            self._spaces = {
                "gemma": {
                    "sum_norm": _norm(sum_vectors_gemma),
                    "sum_dim": sum_vectors_gemma.shape[1],
                    "eng_norm": _norm(eng_vectors),
                    "eng_vocab_map": {w.lower(): i for i, w in enumerate(eng_vocab)},
                }
            }
            self._token_to_idx = {t: i for i, t in enumerate(sum_vocab)}
    return Stub()


def test_displacement_computes_cosine():
    from scripts.analysis.english_displacement import english_displacement

    eng_vocab = ["king"]
    eng_vectors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    sum_vocab = ["lugal"]
    sum_vectors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)  # identical direction

    lookup = _displacement_lookup_stub(eng_vocab, eng_vectors, sum_vocab, sum_vectors)

    result = english_displacement(lookup, "lugal", "king", space="gemma")
    assert result["cosine_similarity"] == pytest.approx(1.0, abs=1e-5)
    assert result["cosine_distance"] == pytest.approx(0.0, abs=1e-5)


def test_displacement_orthogonal_vectors():
    from scripts.analysis.english_displacement import english_displacement

    eng_vocab = ["god"]
    eng_vectors = np.array([[1.0, 0.0]], dtype=np.float32)
    sum_vocab = ["dingir"]
    sum_vectors = np.array([[0.0, 1.0]], dtype=np.float32)
    lookup = _displacement_lookup_stub(eng_vocab, eng_vectors, sum_vocab, sum_vectors)

    result = english_displacement(lookup, "dingir", "god", space="gemma")
    assert result["cosine_similarity"] == pytest.approx(0.0, abs=1e-5)


def test_displacement_raises_on_unknown_sumerian():
    from scripts.analysis.english_displacement import english_displacement

    lookup = _displacement_lookup_stub(
        ["king"], np.eye(1, 2, dtype=np.float32),
        ["lugal"], np.eye(1, 2, dtype=np.float32),
    )
    with pytest.raises(KeyError):
        english_displacement(lookup, "missing", "king", space="gemma")


def test_displacement_raises_on_unknown_english():
    from scripts.analysis.english_displacement import english_displacement

    lookup = _displacement_lookup_stub(
        ["king"], np.eye(1, 2, dtype=np.float32),
        ["lugal"], np.eye(1, 2, dtype=np.float32),
    )
    with pytest.raises(KeyError):
        english_displacement(lookup, "lugal", "unknown", space="gemma")


def test_displacement_is_case_insensitive_on_english():
    from scripts.analysis.english_displacement import english_displacement

    lookup = _displacement_lookup_stub(
        ["king"], np.eye(1, 2, dtype=np.float32),
        ["lugal"], np.eye(1, 2, dtype=np.float32),
    )
    r1 = english_displacement(lookup, "lugal", "king", space="gemma")
    r2 = english_displacement(lookup, "lugal", "KING", space="gemma")
    assert r1["cosine_similarity"] == pytest.approx(r2["cosine_similarity"])
