"""
Generate all cosmogony document figures.

Produces:
  - One narrative-spine UMAP figure (§3 of doc)
  - One heatmap per concept (§3 of each deep dive, §4-8)
  - One synthesis cosmogonic-axis projection (§9)

Usage:
    cd /Users/crashy/Development/cuneiformy
    python scripts/analysis/generate_cosmogony_figures.py

See: docs/superpowers/specs/2026-04-19-sumerian-cosmogony-document-design.md
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent
_LANG_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from languages.sumerian.final_output.sumerian_lookup import SumerianLookup
from languages.sumerian.scripts.analysis.cosmogony_concepts import (
    PRIMARY_CONCEPTS, ANUNNAKI_VOCABULARY, COSMOGONIC_POLES,
)
from framework.analysis.semantic_field import (
    compute_pairwise_distances, render_semantic_field_heatmap,
)
from framework.analysis.umap_projection import umap_cosmogonic_vocabulary

ROOT = _LANG_ROOT
FIG_DIR = ROOT / "docs" / "figures" / "cosmogony"

# Per-concept semantic-field vocabularies: 15-20 thematically-adjacent tokens
# for the heatmap in §3 of each deep dive. Curated; all must pass pre-flight
# vocab check. If any of these are OOV, regenerate concept_slate via pre-flight.
SEMANTIC_FIELDS = {
    "abzu":   ["abzu", "engur", "a", "id", "bad", "kur", "nammu", "enki", "ambar",
               "sirara", "eridu", "dingir", "an", "ki", "bara"],
    "zi":     ["zi", "nam", "nig", "lil", "im", "kalam", "ti", "zid", "lu",
               "dumu", "munus", "lugal", "dingir", "namtar", "gidim"],
    "nam":    ["nam", "nig", "me", "mu", "zid", "ni", "erim", "du", "gal",
               "lugal", "en", "nin", "dumu", "sag", "tuk"],
    "namtar": ["namtar", "nam", "tar", "mu", "zi", "gidim", "kur", "ud", "tag",
               "dingir", "enki", "enlil", "lugal", "zid", "tuk"],
    "me":     ["me", "nam", "nig", "dingir", "enlil", "enki", "an", "inanna",
               "eridu", "kur", "ni", "du", "tum", "sze", "zid"],
}

# Narrative-spine vocabulary labels for the UMAP opener figure.
NARRATIVE_LABELS = {
    "an": "deity", "ki": "deity", "enki": "deity", "enlil": "deity",
    "nammu": "deity", "ninmah": "deity", "inanna": "deity",
    "utu": "deity", "nanna": "deity", "dingir": "deity",
    "lugal": "role",
    "nam": "cosmic_concept", "me": "cosmic_concept", "namtar": "cosmic_concept",
    "zi": "cosmic_concept", "abzu": "cosmic_concept",
    "ima": "matter", "kur": "place",
}


def _load_lookup():
    print("Loading GloVe (~1 min)...")
    glove_vocab, glove_vectors = [], []
    with open(ROOT / "data/processed/glove.6B.300d.txt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(" ")
            glove_vocab.append(parts[0])
            glove_vectors.append([float(x) for x in parts[1:]])
    glove_vectors = np.array(glove_vectors, dtype=np.float32)

    return SumerianLookup(
        gemma_vectors_path=str(ROOT / "final_output/sumerian_aligned_gemma_vectors.npz"),
        glove_vectors_path=str(ROOT / "final_output/sumerian_aligned_vectors.npz"),
        vocab_path=str(ROOT / "final_output/sumerian_aligned_vocab.pkl"),
        gemma_english_path=str(ROOT / "models/english_gemma_whitened_768d.npz"),
        glove_english_vectors=glove_vectors,
        glove_english_vocab=glove_vocab,
    )


def _filter_in_vocab(lookup, tokens: list[str]) -> list[str]:
    known = set(lookup.vocab)
    kept = [t for t in tokens if t in known]
    missing = [t for t in tokens if t not in known]
    if missing:
        print(f"WARN: tokens not in vocab, skipping: {missing}")
    return kept


def _render_concept_heatmap(lookup, concept: dict, out_dir: Path) -> None:
    tag = concept["sumerian"]
    tokens = SEMANTIC_FIELDS.get(tag, [])
    tokens = _filter_in_vocab(lookup, tokens)
    if len(tokens) < 2:
        print(f"WARN: not enough vocab tokens for concept {tag}; skipping heatmap")
        return
    distances = compute_pairwise_distances(lookup, tokens, space="gemma")
    title = f"Semantic field of '{tag}' (Gemma space)"
    out_path = out_dir / f"{tag}_semantic_field_heatmap.png"
    render_semantic_field_heatmap(distances, tokens, title, out_path)
    print(f"Wrote: {out_path}")


def _render_narrative_umap(lookup, out_dir: Path) -> None:
    tokens = _filter_in_vocab(lookup, ANUNNAKI_VOCABULARY)
    umap_cosmogonic_vocabulary(
        lookup, tokens, NARRATIVE_LABELS, space="gemma",
        out_path=out_dir / "anunnaki_narrative_umap.png",
        title="Anunnaki and cosmogonic vocabulary (Gemma 2D projection)",
    )
    print(f"Wrote: {out_dir / 'anunnaki_narrative_umap.png'}")


def _render_axis_projection(lookup, out_dir: Path) -> None:
    """Project concepts onto a cosmogonic axis (primordial -> decree)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gemma_space = lookup._spaces["gemma"]
    idx_map = {t: i for i, t in enumerate(lookup.vocab)}

    def _centroid(tokens):
        tokens_present = [t for t in tokens if t in idx_map]
        if not tokens_present:
            return None
        vecs = gemma_space["sum_norm"][[idx_map[t] for t in tokens_present]]
        c = vecs.mean(axis=0)
        n = np.linalg.norm(c)
        return c / n if n > 0 else c

    prim = _centroid(COSMOGONIC_POLES["primordial_pole"])
    decree = _centroid(COSMOGONIC_POLES["decree_pole"])
    axis = decree - prim
    axis_norm = axis / (np.linalg.norm(axis) or 1.0)

    concept_labels, projections = [], []
    for concept in PRIMARY_CONCEPTS:
        tag = concept["sumerian"]
        if tag in idx_map:
            v = gemma_space["sum_norm"][idx_map[tag]]
            proj = float(np.dot(v, axis_norm))
            concept_labels.append(tag)
            projections.append(proj)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axhline(0, color="lightgray", lw=1)
    ax.scatter(projections, [0] * len(projections), s=120, zorder=3)
    for x, lbl in zip(projections, concept_labels):
        ax.annotate(lbl, (x, 0), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=11)
    ax.set_title("Cosmogonic axis: primordial (←) to decree (→)")
    ax.set_xlabel("projection on axis (primordial pole − decree pole)")
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_dir / "cosmogony_axis_projection.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out_dir / 'cosmogony_axis_projection.png'}")


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    lookup = _load_lookup()

    _render_narrative_umap(lookup, FIG_DIR)
    for concept in PRIMARY_CONCEPTS:
        _render_concept_heatmap(lookup, concept, FIG_DIR)
    _render_axis_projection(lookup, FIG_DIR)

    return 0


if __name__ == "__main__":
    sys.exit(main())
