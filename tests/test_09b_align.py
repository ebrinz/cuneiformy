import numpy as np


def test_align_09b_shape_contract_at_768d():
    """09b must work when the English target dim is 768 (EmbeddingGemma)."""
    from scripts.align_09b import build_training_data, train_ridge, evaluate_alignment

    anchors = [
        {"sumerian": "lugal", "english": "king"},
        {"sumerian": "e2", "english": "house"},
        {"sumerian": "dingir", "english": "god"},
    ]

    sum_vocab = {"lugal": 0, "e2": 1, "dingir": 2}
    sum_vectors = np.random.randn(3, 1536).astype(np.float32)

    eng_vocab = {"king": 0, "house": 1, "god": 2}
    eng_vectors = np.random.randn(3, 768).astype(np.float32)

    X, Y, valid = build_training_data(
        anchors, sum_vocab, sum_vectors, eng_vocab, eng_vectors
    )
    assert X.shape == (3, 1536)
    assert Y.shape == (3, 768)

    model = train_ridge(X, Y, alpha=100)
    assert model.coef_.shape == (768, 1536)

    Y_pred = model.predict(X)
    results = evaluate_alignment(
        Y_pred,
        ["king", "house", "god"],
        ["king", "house", "god"],
        eng_vectors,
        ks=(1, 2, 3),
    )
    assert "top1" in results
    assert "top2" in results
    assert "top3" in results
