import os
import tempfile
import pytest


def test_corpus_iterator():
    """CorpusIterator should yield lines from a text file."""
    from scripts.fasttext_07 import CorpusIterator

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("lugal e2 gal\n")
        f.write("dingir an ki\n")
        f.write("mu sar ra\n")
        f.flush()

        lines = list(CorpusIterator(f.name))

    os.unlink(f.name)

    assert len(lines) == 3
    assert lines[0] == ["lugal", "e2", "gal"]
    assert lines[1] == ["dingir", "an", "ki"]


def test_train_fasttext_model():
    """Train a small FastText model and verify dimensions."""
    from scripts.fasttext_07 import train_fasttext

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for _ in range(100):
            f.write("lugal e2 gal dingir an ki mu sar ra nam\n")
        f.flush()

        with tempfile.TemporaryDirectory() as tmpdir:
            model = train_fasttext(
                corpus_path=f.name,
                output_dir=tmpdir,
                vector_size=32,
                window=5,
                min_count=1,
                epochs=2,
            )

            assert model.vector_size == 32
            assert "lugal" in model.wv

    os.unlink(f.name)
