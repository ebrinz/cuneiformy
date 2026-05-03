"""
Qualitative concept-cluster comparison: GloVe-aligned vs Gemma-aligned
Sumerian -> English reverse queries.

For each seed English word (drawn from docs/NEAR_TERM_STRATEGY.md):
  1. Find top-10 Sumerian nearest neighbors (in the given alignment space)
  2. For each of those, find top-5 English nearest neighbors in the same space
  3. Dump as a Markdown section

The report is read by a human to judge which space produces more coherent
concept clusters. No automated pass/fail.

See: docs/superpowers/specs/2026-04-16-gemma-embed-alignment-design.md
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FINAL_OUTPUT_DIR = ROOT / "final_output"
DATA_PROCESSED = ROOT / "data" / "processed"

# Seed words from docs/NEAR_TERM_STRATEGY.md Phase 1 concept domains.
SEED_WORDS = {
    "creation": ["create", "begin", "birth", "origin", "emerge", "form", "separate"],
    "fate_meaning": ["fate", "destiny", "purpose", "decree", "name", "order"],
    "self_soul": ["self", "soul", "spirit", "mind", "heart", "breath", "shadow"],
}

K_SUMERIAN = 10
K_ENGLISH_REPROJECTION = 5

GEMMA_CACHES = {
    "gloss": MODELS_DIR / "english_gemma_768d.npz",
    "bare": MODELS_DIR / "english_gemma_bare_768d.npz",
    "whitened": MODELS_DIR / "english_gemma_whitened_768d.npz",
    "bare_whitened": MODELS_DIR / "english_gemma_bare_whitened_768d.npz",
}
RIDGE_WEIGHT_PATHS = {
    "gloss": MODELS_DIR / "ridge_weights_gemma.npz",
    "bare": MODELS_DIR / "ridge_weights_gemma_bare.npz",
    "whitened": MODELS_DIR / "ridge_weights_gemma_whitened.npz",
    "bare_whitened": MODELS_DIR / "ridge_weights_gemma_bare_whitened.npz",
}


def load_glove_space():
    """Return (eng_vocab, eng_vectors, sum_vocab, sum_aligned_vectors) for GloVe space."""
    glove_path = DATA_PROCESSED / "glove.6B.300d.txt"
    eng_vocab, eng_vec_list = [], []
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ")
            eng_vocab.append(parts[0])
            eng_vec_list.append(np.array([float(x) for x in parts[1:]], dtype=np.float32))
    eng_vectors = np.array(eng_vec_list)

    aligned = np.load(str(FINAL_OUTPUT_DIR / "sumerian_aligned_vectors.npz"))
    sum_aligned = aligned["vectors"].astype(np.float32)
    with open(FINAL_OUTPUT_DIR / "sumerian_aligned_vocab.pkl", "rb") as f:
        sum_vocab = [str(w) for w in pickle.load(f)]

    return eng_vocab, eng_vectors, sum_vocab, sum_aligned


def load_gemma_space(mode: str = "gloss"):
    """Return (eng_vocab, eng_vectors, sum_vocab, sum_aligned_vectors) for Gemma space."""
    gemma = np.load(str(GEMMA_CACHES[mode]))
    eng_vocab = [str(w) for w in gemma["vocab"]]
    eng_vectors = gemma["vectors"].astype(np.float32)

    fused = np.load(str(MODELS_DIR / "fused_embeddings_1536d.npz"))
    sum_vocab = [str(w) for w in fused["vocab"]]
    sum_fused = fused["vectors"].astype(np.float32)

    ridge = np.load(str(RIDGE_WEIGHT_PATHS[mode]))
    coef = ridge["coef"]
    intercept = ridge["intercept"]
    sum_aligned = sum_fused @ coef.T + intercept

    return eng_vocab, eng_vectors, sum_vocab, sum_aligned.astype(np.float32)


def cosine_topk(query: np.ndarray, candidates: np.ndarray, k: int) -> list[int]:
    """Return indices of top-k cosine-nearest rows in candidates to query."""
    q = query / (np.linalg.norm(query) + 1e-12)
    c_norms = np.linalg.norm(candidates, axis=1, keepdims=True)
    c_norms[c_norms == 0] = 1
    c_norm = candidates / c_norms
    sims = c_norm @ q
    return list(np.argsort(-sims)[:k])


def reverse_query(
    seed: str,
    eng_vocab: list[str],
    eng_vectors: np.ndarray,
    sum_vocab: list[str],
    sum_aligned: np.ndarray,
    k_sum: int,
    k_eng: int,
) -> dict:
    """Run the reverse-query pattern for one English seed word.

    English seed -> top-k_sum Sumerian neighbors -> for each, top-k_eng English neighbors.
    Returns a dict with the seed, its top Sumerian neighbors, and their
    English re-projections. Returns an error-marked result if seed missing from eng_vocab.
    """
    if seed not in eng_vocab:
        return {"seed": seed, "error": "seed not in English vocab", "sumerian_neighbors": []}

    eng_idx = eng_vocab.index(seed)
    seed_vec = eng_vectors[eng_idx]

    sum_top = cosine_topk(seed_vec, sum_aligned, k_sum)
    neighbors = []
    for s_idx in sum_top:
        s_word = sum_vocab[s_idx]
        s_vec = sum_aligned[s_idx]
        eng_top = cosine_topk(s_vec, eng_vectors, k_eng)
        neighbors.append(
            {
                "sumerian": s_word,
                "english_reprojection": [eng_vocab[i] for i in eng_top],
            }
        )
    return {"seed": seed, "sumerian_neighbors": neighbors}


def format_cluster_markdown(glove_result: dict, gemma_result: dict) -> str:
    """Render a side-by-side Markdown section for one seed word."""
    seed = glove_result["seed"]
    lines = [f"### `{seed}`", ""]

    if glove_result.get("error") or gemma_result.get("error"):
        g_err = glove_result.get("error", "")
        m_err = gemma_result.get("error", "")
        lines.append(f"- GloVe: {g_err or 'OK'}")
        lines.append(f"- Gemma: {m_err or 'OK'}")
        lines.append("")
        return "\n".join(lines)

    lines.append("| # | GloVe: Sumerian -> English re-projection | Gemma: Sumerian -> English re-projection |")
    lines.append("|---|---|---|")
    rows = max(len(glove_result["sumerian_neighbors"]), len(gemma_result["sumerian_neighbors"]))
    for i in range(rows):
        g = glove_result["sumerian_neighbors"][i] if i < len(glove_result["sumerian_neighbors"]) else None
        m = gemma_result["sumerian_neighbors"][i] if i < len(gemma_result["sumerian_neighbors"]) else None
        g_str = (
            f"**{g['sumerian']}** -> {', '.join(g['english_reprojection'])}"
            if g else ""
        )
        m_str = (
            f"**{m['sumerian']}** -> {', '.join(m['english_reprojection'])}"
            if m else ""
        )
        lines.append(f"| {i+1} | {g_str} | {m_str} |")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Qualitative concept-cluster comparison: GloVe vs Gemma.")
    parser.add_argument(
        "--gemma-mode",
        choices=list(GEMMA_CACHES.keys()),
        default="gloss",
        help="Which Gemma variant to compare against GloVe.",
    )
    args = parser.parse_args()
    mode = args.gemma_mode
    report_path = RESULTS_DIR / f"concept_clusters_comparison_{mode}.md"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    required = [
        FINAL_OUTPUT_DIR / "sumerian_aligned_vectors.npz",
        FINAL_OUTPUT_DIR / "sumerian_aligned_vocab.pkl",
        GEMMA_CACHES[mode],
        RIDGE_WEIGHT_PATHS[mode],
        MODELS_DIR / "fused_embeddings_1536d.npz",
        DATA_PROCESSED / "glove.6B.300d.txt",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        for p in missing:
            print(f"ERROR: missing required artifact: {p}", file=sys.stderr)
        sys.exit(1)

    print("Loading GloVe space...")
    g_eng_vocab, g_eng_vec, g_sum_vocab, g_sum_aligned = load_glove_space()
    print(f"GloVe: {len(g_eng_vocab)} English, {len(g_sum_vocab)} Sumerian aligned")

    print(f"Loading Gemma {mode} space...")
    m_eng_vocab, m_eng_vec, m_sum_vocab, m_sum_aligned = load_gemma_space(mode)
    print(f"Gemma ({mode}): {len(m_eng_vocab)} English, {len(m_sum_vocab)} Sumerian aligned")

    sections = []
    sections.append("# Concept Cluster Comparison: GloVe vs EmbeddingGemma")
    sections.append("")
    sections.append(
        "Reverse-query reading: English seed -> top-10 Sumerian nearest neighbors, "
        "then for each Sumerian word, top-5 English nearest neighbors in the same space."
    )
    sections.append("")
    sections.append(
        "Human-read qualitative gate for phase A. The goal is to judge which space "
        "produces more semantically coherent clusters for the concept domains in "
        "`docs/NEAR_TERM_STRATEGY.md`."
    )
    sections.append("")

    for domain, words in SEED_WORDS.items():
        sections.append(f"## Domain: {domain}")
        sections.append("")
        for word in words:
            g_res = reverse_query(
                word, g_eng_vocab, g_eng_vec, g_sum_vocab, g_sum_aligned,
                K_SUMERIAN, K_ENGLISH_REPROJECTION,
            )
            m_res = reverse_query(
                word, m_eng_vocab, m_eng_vec, m_sum_vocab, m_sum_aligned,
                K_SUMERIAN, K_ENGLISH_REPROJECTION,
            )
            sections.append(format_cluster_markdown(g_res, m_res))

    report_path.write_text("\n".join(sections))
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
