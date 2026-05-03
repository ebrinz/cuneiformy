import numpy as np
import pytest


def test_build_training_data():
    """Build X (Sumerian) and Y (English) matrices from anchors."""
    from languages.sumerian.scripts.align_09 import build_training_data

    anchors = [
        {"sumerian": "lugal", "english": "king"},
        {"sumerian": "e2", "english": "house"},
        {"sumerian": "unknown", "english": "missing"},
    ]

    sum_vocab = {"lugal": 0, "e2": 1, "dingir": 2}
    sum_vectors = np.random.randn(3, 1536).astype(np.float32)

    eng_vocab = {"king": 0, "house": 1, "water": 2}
    eng_vectors = np.random.randn(3, 300).astype(np.float32)

    X, Y, valid_anchors = build_training_data(
        anchors, sum_vocab, sum_vectors, eng_vocab, eng_vectors
    )

    assert X.shape == (2, 1536)
    assert Y.shape == (2, 300)
    assert len(valid_anchors) == 2


def test_evaluate_alignment():
    """Evaluate Top-K accuracy of alignment."""
    from languages.sumerian.scripts.align_09 import evaluate_alignment

    np.random.seed(42)
    n_test = 10
    dim = 300

    Y_test = np.random.randn(n_test, dim).astype(np.float32)
    Y_pred = Y_test + np.random.randn(n_test, dim).astype(np.float32) * 0.01

    glove_vocab = [f"word_{i}" for i in range(n_test + 50)]
    glove_vectors = np.vstack([
        Y_test,
        np.random.randn(50, dim).astype(np.float32),
    ])

    test_english = [f"word_{i}" for i in range(n_test)]

    results = evaluate_alignment(Y_pred, test_english, glove_vocab, glove_vectors)

    assert "top1" in results
    assert "top5" in results
    assert "top10" in results
    assert results["top1"] > 0.5


def test_train_ridge():
    """Train Ridge regression and verify it produces correct output shape."""
    from languages.sumerian.scripts.align_09 import train_ridge

    X = np.random.randn(100, 1536).astype(np.float32)
    Y = np.random.randn(100, 300).astype(np.float32)

    model = train_ridge(X, Y, alpha=0.001)

    Y_pred = model.predict(X[:5])
    assert Y_pred.shape == (5, 300)
