"""
End-to-end integration test using synthetic data.
Verifies the full pipeline: corpus -> embeddings -> fusion -> alignment -> lookup.
"""
import json
import numpy as np
import os
import tempfile
import pytest


def test_full_pipeline_synthetic():
    """Run the full pipeline on tiny synthetic data."""
    from scripts.clean_05 import clean_atf_line, build_corpus
    from scripts.anchors_06 import extract_epsd2_anchors, merge_anchors
    from scripts.fasttext_07 import train_fasttext
    from scripts.fuse_08 import fuse_embeddings
    from scripts.align_09 import build_training_data, train_ridge, evaluate_alignment
    from scripts.export_10 import project_all_vectors

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Create synthetic corpus
        words = ["lugal", "e2", "dingir", "an", "ki", "mu", "nam", "en", "gal", "nin",
                 "sar", "kur", "igi", "sag", "du", "gin", "ba", "bi", "na", "zu"]
        np.random.seed(42)
        corpus_lines = []
        for _ in range(500):
            n = np.random.randint(3, 10)
            line = " ".join(np.random.choice(words, n))
            corpus_lines.append(line)

        corpus_path = os.path.join(tmpdir, "corpus.txt")
        with open(corpus_path, "w") as f:
            for line in corpus_lines:
                f.write(line + "\n")

        # 2. Train FastText (small)
        model = train_fasttext(
            corpus_path=corpus_path,
            output_dir=tmpdir,
            vector_size=32,
            window=5,
            min_count=1,
            epochs=5,
        )
        assert len(model.wv) >= 15

        # 3. Fuse with zero padding
        vocab = list(model.wv.index_to_key)
        text_vecs = np.array([model.wv[w] for w in vocab], dtype=np.float32)
        fused, fused_vocab = fuse_embeddings(vocab, text_vecs, pad_dim=32)
        assert fused.shape[1] == 64

        # 4. Create synthetic GloVe (tiny)
        glove_words = ["king", "house", "god", "heaven", "earth", "name", "fate",
                       "lord", "great", "queen", "write", "mountain", "eye", "head",
                       "go", "walk", "give", "this", "that", "know"]
        glove_vecs = np.random.randn(len(glove_words), 16).astype(np.float32)
        eng_vocab = {w: i for i, w in enumerate(glove_words)}

        # 5. Create anchors
        anchors = [
            {"sumerian": "lugal", "english": "king", "confidence": 0.95, "source": "ePSD2"},
            {"sumerian": "e2", "english": "house", "confidence": 0.90, "source": "ePSD2"},
            {"sumerian": "dingir", "english": "god", "confidence": 0.85, "source": "ePSD2"},
            {"sumerian": "an", "english": "heaven", "confidence": 0.80, "source": "ePSD2"},
            {"sumerian": "ki", "english": "earth", "confidence": 0.80, "source": "ePSD2"},
            {"sumerian": "mu", "english": "name", "confidence": 0.75, "source": "ePSD2"},
            {"sumerian": "nam", "english": "fate", "confidence": 0.70, "source": "ePSD2"},
            {"sumerian": "en", "english": "lord", "confidence": 0.70, "source": "ePSD2"},
            {"sumerian": "gal", "english": "great", "confidence": 0.65, "source": "ePSD2"},
            {"sumerian": "nin", "english": "queen", "confidence": 0.60, "source": "ePSD2"},
        ]

        sum_vocab_dict = {w: i for i, w in enumerate(fused_vocab)}

        # 6. Build training data
        X, Y, valid = build_training_data(anchors, sum_vocab_dict, fused, eng_vocab, glove_vecs)
        assert len(valid) == 10

        # 7. Train Ridge
        ridge = train_ridge(X, Y, alpha=0.001)
        Y_pred = ridge.predict(X)
        assert Y_pred.shape == (10, 16)

        # 8. Project all vectors
        projected = project_all_vectors(fused, ridge.coef_, ridge.intercept_)
        assert projected.shape == (len(fused_vocab), 16)
        assert projected.dtype == np.float16
