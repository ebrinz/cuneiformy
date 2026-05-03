"""
One-shot precompute: EmbeddingGemma vectors for GloVe's 400k English vocab.

Used by phase A of the Gemma alignment benchmark. Output is cached to
models/english_gemma_768d.npz and reused by 09b_align_gemma.py.

See: docs/superpowers/specs/2026-04-16-gemma-embed-alignment-design.md
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from nltk.corpus import wordnet

GEMMA_MODEL = "google/embeddinggemma-300m"
BATCH_SIZE = 64
EMBED_DIM = 768

ROOT = Path(__file__).parent.parent
_REPO_ROOT = ROOT.parent
GLOVE_PATH = _REPO_ROOT / "languages" / "sumerian" / "data" / "processed" / "glove.6B.300d.txt"
GLOSS_OUTPUT_PATH = ROOT / "models" / "english_gemma_768d.npz"
BARE_OUTPUT_PATH = ROOT / "models" / "english_gemma_bare_768d.npz"


def format_gloss(word: str, definition: str | None) -> str:
    """Format a word + definition pair for EmbeddingGemma encoding.

    Hit: "word: definition text"
    Miss (None or empty): "word"
    """
    if definition:
        return f"{word}: {definition}"
    return word


def lookup_gloss(word: str) -> str | None:
    """Look up the first WordNet synset's definition for a word.

    Returns None if the word has no WordNet entry.
    """
    synsets = wordnet.synsets(word)
    if not synsets:
        return None
    return synsets[0].definition()


def load_glove_vocab(glove_path: Path) -> list[str]:
    """Read only the first column (word) of the GloVe text file.

    Skips vectors entirely — we just want the vocabulary list.
    """
    vocab = []
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            word = line.split(" ", 1)[0]
            vocab.append(word)
    return vocab


def output_is_up_to_date(output_path: Path, expected_vocab: list[str]) -> bool:
    """Return True if output file exists and its vocab matches expected_vocab exactly."""
    if not output_path.exists():
        return False
    try:
        cached = np.load(str(output_path))
        cached_vocab = [str(w) for w in cached["vocab"]]
        return cached_vocab == expected_vocab
    except (KeyError, ValueError, OSError):
        return False


def encode_batch_with_retry(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
) -> np.ndarray:
    """Encode a list of texts with one-shot halved-batch retry on failure."""
    try:
        return model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            prompt_name="Retrieval-document",
        )
    except (RuntimeError, MemoryError):
        if batch_size <= 1:
            raise
        return model.encode(
            texts,
            batch_size=max(1, batch_size // 2),
            convert_to_numpy=True,
            show_progress_bar=False,
            prompt_name="Retrieval-document",
        )


def main():
    parser = argparse.ArgumentParser(description="Precompute EmbeddingGemma vectors for GloVe vocab.")
    parser.add_argument(
        "--bare",
        action="store_true",
        help="Skip WordNet glosses — encode each word as bare text. Retry after gloss run underperformed.",
    )
    args = parser.parse_args()
    use_gloss = not args.bare
    output_path = GLOSS_OUTPUT_PATH if use_gloss else BARE_OUTPUT_PATH
    mode_label = "gloss" if use_gloss else "bare"

    if not GLOVE_PATH.exists():
        print(f"ERROR: GloVe file not found at {GLOVE_PATH}", file=sys.stderr)
        print("Run scripts/download_glove.py first.", file=sys.stderr)
        sys.exit(1)

    if use_gloss:
        try:
            wordnet.synsets("test")
        except LookupError:
            print("ERROR: WordNet data not installed.", file=sys.stderr)
            print("Run: python -c \"import nltk; nltk.download('wordnet')\"", file=sys.stderr)
            sys.exit(1)

    print(f"Mode: {mode_label}")
    print(f"Loading GloVe vocab from {GLOVE_PATH}")
    vocab = load_glove_vocab(GLOVE_PATH)
    print(f"GloVe vocab: {len(vocab)} words")

    if output_is_up_to_date(output_path, vocab):
        print(f"Output already up-to-date at {output_path}, skipping.")
        sys.exit(0)

    if use_gloss:
        print("Looking up WordNet glosses...")
        glosses = []
        hits = 0
        for word in tqdm(vocab, desc="WordNet"):
            definition = lookup_gloss(word)
            if definition:
                hits += 1
            glosses.append(format_gloss(word, definition))
        hit_rate = hits / len(vocab) * 100
        print(f"WordNet hits: {hits}/{len(vocab)} ({hit_rate:.1f}%)")
    else:
        print("Bare-word mode: skipping WordNet lookup.")
        glosses = list(vocab)
        hit_rate = 0.0

    print(f"Loading {GEMMA_MODEL}...")
    model = SentenceTransformer(GEMMA_MODEL)
    print(f"Encoder device: {model.device}")

    print(f"Encoding {len(glosses)} texts at batch_size={BATCH_SIZE}...")
    all_vectors = np.zeros((len(glosses), EMBED_DIM), dtype=np.float32)
    for start in tqdm(range(0, len(glosses), BATCH_SIZE), desc="Encode"):
        end = min(start + BATCH_SIZE, len(glosses))
        batch = glosses[start:end]
        vecs = encode_batch_with_retry(model, batch, BATCH_SIZE)
        all_vectors[start:end] = vecs.astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(output_path),
        vocab=np.array(vocab),
        vectors=all_vectors,
        gloss_hit_rate=np.float32(hit_rate),
        gemma_model=np.array(GEMMA_MODEL),
        mode=np.array(mode_label),
    )
    print(f"Saved {all_vectors.shape} to {output_path}")
    if use_gloss:
        print(f"Gloss hit rate: {hit_rate:.2f}%")


if __name__ == "__main__":
    main()
