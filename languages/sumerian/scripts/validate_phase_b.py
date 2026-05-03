"""
Phase B validation: end-to-end sanity check on the dual-view pipeline.

Loads the exported artifacts, queries several seed words in both spaces,
regenerates the concept-cluster comparison report, and asserts all outputs
look healthy.

Exits 0 on success, 1 with a specific error on failure.

See: docs/superpowers/specs/2026-04-16-phase-b-gemma-downstream-design.md
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
_REPO_ROOT = Path(__file__).parent.parent.parent.parent
# Ensure repo root is importable when invoked directly (pytest.ini only affects pytest).
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

FINAL_OUTPUT = ROOT / "final_output"
MODELS_DIR = ROOT / "models"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

SEED_WORDS = ("king", "fate", "soul", "heart", "name")
EXPECTED_REPORT = RESULTS_DIR / "concept_clusters_comparison_whitened.md"
EXPECTED_DOMAIN_HEADERS = (
    "## Domain: creation",
    "## Domain: fate_meaning",
    "## Domain: self_soul",
)


def _load_glove_vectors() -> tuple[np.ndarray, list[str]]:
    glove_path = DATA_PROCESSED / "glove.6B.300d.txt"
    vocab: list[str] = []
    vec_list: list[np.ndarray] = []
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(" ")
            vocab.append(parts[0])
            vec_list.append(np.array([float(x) for x in parts[1:]], dtype=np.float32))
    return np.array(vec_list), vocab


def main() -> int:
    print("=== Phase B validation ===\n")

    from languages.sumerian.final_output.sumerian_lookup import SumerianLookup

    print("Loading GloVe (400k, ~1 min)...")
    glove_eng_vec, glove_eng_vocab = _load_glove_vectors()
    print(f"GloVe: {glove_eng_vec.shape}")

    print("Building SumerianLookup...")
    lookup = SumerianLookup(
        gemma_vectors_path=str(FINAL_OUTPUT / "sumerian_aligned_gemma_vectors.npz"),
        glove_vectors_path=str(FINAL_OUTPUT / "sumerian_aligned_vectors.npz"),
        vocab_path=str(FINAL_OUTPUT / "sumerian_aligned_vocab.pkl"),
        gemma_english_path=str(MODELS_DIR / "english_gemma_whitened_768d.npz"),
        glove_english_vectors=glove_eng_vec,
        glove_english_vocab=glove_eng_vocab,
    )

    print(f"\nSeed queries (top-5 per space):")
    for word in SEED_WORDS:
        both = lookup.find_both(word, top_k=5)
        if not both["gemma"] or not both["glove"]:
            print(f"ERROR: empty results for {word!r}", file=sys.stderr)
            return 1
        gemma_str = ", ".join(f"{w}({s:.2f})" for w, s in both["gemma"])
        glove_str = ", ".join(f"{w}({s:.2f})" for w, s in both["glove"])
        print(f"  {word:>8s}")
        print(f"    gemma: {gemma_str}")
        print(f"    glove: {glove_str}")

    print(f"\nAnalogy (king is to queen as father is to ?):")
    gemma_a = lookup.find_analogy("king", "queen", "father", top_k=3, space="gemma")
    glove_a = lookup.find_analogy("king", "queen", "father", top_k=3, space="glove")
    if not gemma_a or not glove_a:
        print("ERROR: empty analogy results", file=sys.stderr)
        return 1
    print(f"  gemma: {[w for w, _ in gemma_a]}")
    print(f"  glove: {[w for w, _ in glove_a]}")

    print(f"\nRegenerating concept-cluster comparison (whitened Gemma vs GloVe)...")
    result = subprocess.run(
        [sys.executable, "scripts/evaluate_concept_clusters.py", "--gemma-mode", "whitened"],
        cwd=str(ROOT),
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("ERROR: evaluate_concept_clusters.py failed", file=sys.stderr)
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        return 1
    lines = result.stdout.strip().splitlines()
    if lines:
        print(lines[-1])

    if not EXPECTED_REPORT.exists():
        print(f"ERROR: expected report at {EXPECTED_REPORT} — not found", file=sys.stderr)
        return 1
    report_text = EXPECTED_REPORT.read_text()
    for header in EXPECTED_DOMAIN_HEADERS:
        if header not in report_text:
            print(f"ERROR: report missing header {header!r}", file=sys.stderr)
            return 1

    print("\n=== Phase B validation PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
