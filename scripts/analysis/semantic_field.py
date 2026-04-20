"""
Semantic-field pairwise distance + heatmap rendering.

Used per-concept in the Sumerian cosmogony document (§3 of each deep dive).
Takes a list of thematically-adjacent Sumerian tokens, computes pairwise
cosine distances via a SumerianLookup's pre-normalized vectors, and renders
a matplotlib heatmap PNG for commit.

See: docs/superpowers/specs/2026-04-19-sumerian-cosmogony-document-design.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def compute_pairwise_distances(
    lookup,
    sumerian_tokens: list[str],
    space: str = "gemma",
) -> np.ndarray:
    """N x N cosine-distance matrix over the given Sumerian tokens.

    Distance = 1 - cosine-similarity. Range [0, 2], symmetric, diagonal = 0.
    """
    s = lookup._spaces[space]
    sum_norm = s["sum_norm"]
    vocab = lookup.vocab

    idx_map = {t: i for i, t in enumerate(vocab)}
    indices = []
    for tok in sumerian_tokens:
        if tok not in idx_map:
            raise KeyError(f"unknown Sumerian token: {tok!r}")
        indices.append(idx_map[tok])

    rows = sum_norm[indices]  # (n, dim), already L2-normalized
    sims = rows @ rows.T  # cosine similarities
    sims = np.clip(sims, -1.0, 1.0)
    return 1.0 - sims


def render_semantic_field_heatmap(
    distances: np.ndarray,
    tokens: list[str],
    title: str,
    out_path: Path,
) -> None:
    """Render pairwise-distance matrix as a matplotlib heatmap PNG."""
    import matplotlib
    matplotlib.use("Agg")  # non-interactive; required for CI / headless runs
    import matplotlib.pyplot as plt

    n = len(tokens)
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * n), max(6, 0.5 * n)))

    im = ax.imshow(distances, cmap="viridis_r", aspect="auto", vmin=0, vmax=2)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tokens, rotation=45, ha="right")
    ax.set_yticklabels(tokens)
    ax.set_title(title)

    fig.colorbar(im, ax=ax, label="cosine distance (1 - cos sim)")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
