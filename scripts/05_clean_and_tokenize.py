"""
Corpus Cleaning: Convert ATF transliterations to clean, tokenized text for FastText.

ATF cleanup:
- Strip editorial markers: [...], (...), #, !, ?
- Remove determinatives notation: {ki}, {d}, etc. -> keep the content
- Remove compound sign notations: |...|, <...>
- Remove semantic/language markers: _..._
- Normalize transliteration conventions (sz->c, etc.) to match ORACC anchors
- Replace hyphens with spaces (morpheme boundaries)
- Normalize subscript digits to ASCII
- Normalize whitespace

Output: cleaned_corpus.txt (one line per text, space-separated tokens)
"""
import json
import re
from pathlib import Path

DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"

# Unicode subscript -> ASCII digit mapping
_SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

# ATF-to-ORACC transliteration normalization
_TRANSLIT_REPLACEMENTS = [
    ("sz", "c"),      # shin: sz (ATF) -> c (ORACC)
    ("s,", "s"),      # tsade: s, (ATF) -> s (ORACC) - note: must come before generic comma strip
    ("t,", "t"),      # emphatic t
    ("SZ", "C"),      # uppercase
]


def normalize_transliteration(word: str) -> str:
    """Normalize ATF transliteration conventions to match ORACC citation forms."""
    # Convert subscript digits to ASCII
    word = word.translate(_SUBSCRIPT_MAP)

    # ATF -> ORACC convention mapping
    for old, new in _TRANSLIT_REPLACEMENTS:
        word = word.replace(old, new)

    return word.lower()


def clean_atf_line(line: str) -> str:
    """Clean a single ATF transliteration line."""
    if line.startswith("$"):
        return ""

    # Remove correction notations: word!(SIGN) -> word
    line = re.sub(r"!\([^)]*\)", "", line)

    # Remove square brackets but keep content
    line = line.replace("[", "").replace("]", "")

    # Remove angle brackets (uncertain readings) but keep content
    line = re.sub(r"<([^>]*)>", r"\1", line)

    # Remove pipe-delimited compound signs entirely: |GA2xAN|, |KI.AN|
    line = re.sub(r"\|[^|]*\|", "", line)
    # Also handle partial pipes at word boundaries
    line = re.sub(r"\|[A-Z0-9.x+]+", "", line)
    line = re.sub(r"[A-Z0-9.x+]+\|", "", line)

    # Remove semantic/language markers: _d_, _X_, _dumu_
    line = re.sub(r"_[^_\s]*_", "", line)

    # Remove parentheses when not preceded by digit (preserve number notations)
    line = re.sub(r"(?<!\d)\(([^)]*)\)", r"\1", line)

    # Remove damage marker #
    line = line.replace("#", "")

    # Remove uncertainty marker ?
    line = line.replace("?", "")

    # Remove exclamation (correction marker)
    line = line.replace("!", "")

    # Handle determinatives {ki}, {d}, etc. - keep the content
    line = re.sub(r"\{([^}]*)\}", r" \1 ", line)

    # Replace hyphens with spaces (morpheme boundaries)
    line = line.replace("-", " ")

    # Replace dots used as separators (but not in numbers like 1.0)
    line = re.sub(r"(?<!\d)\.(?!\d)", " ", line)

    # Normalize whitespace
    line = re.sub(r"\s+", " ", line).strip()

    if not line or line == "...":
        return ""

    # Normalize each token's transliteration
    tokens = line.split()
    normalized = []
    for tok in tokens:
        tok = normalize_transliteration(tok)
        # Skip pure numbers, single chars that are likely noise, ALL-CAPS (sign names)
        if tok and not tok.isupper() and len(tok) > 0:
            normalized.append(tok)

    return " ".join(normalized)


def build_corpus(texts: list[dict]) -> list[str]:
    """
    Build cleaned corpus lines from merged texts.

    Each text's lines are cleaned and joined into a single line,
    matching the one-sentence-per-line format FastText expects.
    """
    corpus_lines = []
    for text in texts:
        cleaned_words = []
        for line in text.get("lines", []):
            cleaned = clean_atf_line(line)
            if cleaned:
                cleaned_words.append(cleaned)
        if cleaned_words:
            corpus_lines.append(" ".join(cleaned_words))
    return corpus_lines


def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    merged_path = DATA_PROCESSED / "merged_corpus.json"
    with open(merged_path) as f:
        texts = json.load(f)

    corpus_lines = build_corpus(texts)

    output_path = DATA_PROCESSED / "cleaned_corpus.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        for line in corpus_lines:
            f.write(line + "\n")

    total_tokens = sum(len(line.split()) for line in corpus_lines)
    vocab = set()
    for line in corpus_lines:
        vocab.update(line.split())

    print(f"Corpus lines: {len(corpus_lines)}")
    print(f"Total tokens: {total_tokens}")
    print(f"Unique tokens: {len(vocab)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
