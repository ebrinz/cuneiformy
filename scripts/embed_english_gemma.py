"""
One-shot precompute: EmbeddingGemma vectors for GloVe's 400k English vocab.

Used by phase A of the Gemma alignment benchmark. Output is cached to
models/english_gemma_768d.npz and reused by 09b_align_gemma.py.

See: docs/superpowers/specs/2026-04-16-gemma-embed-alignment-design.md
"""
from nltk.corpus import wordnet


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
