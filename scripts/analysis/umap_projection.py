"""
UMAP 2D projection for the narrative-spine opener and synthesis figures.

Falls back to PCA for vocabularies too small for UMAP (n < 15 by default).
Deterministic with fixed seed.

See: docs/superpowers/specs/2026-04-19-sumerian-cosmogony-document-design.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


UMAP_MIN_VOCAB = 15
DEFAULT_SEED = 42


def _compute_embedding(normalized_vectors: np.ndarray, seed: int = DEFAULT_SEED) -> np.ndarray:
    """Compute 2D embedding. Uses UMAP if vocab big enough, else PCA."""
    n = normalized_vectors.shape[0]
    if n >= UMAP_MIN_VOCAB:
        import umap
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(15, n - 1),
            metric="cosine",
            random_state=seed,
        )
        return reducer.fit_transform(normalized_vectors)
    # PCA fallback.
    centered = normalized_vectors - normalized_vectors.mean(axis=0)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt.T[:, :2]


def umap_cosmogonic_vocabulary(
    lookup,
    tokens: list[str],
    labels: dict[str, str],
    space: str = "gemma",
    out_path: Path = None,
    title: str = "Cosmogonic vocabulary (2D projection)",
    seed: int = DEFAULT_SEED,
) -> None:
    """Project the given Sumerian tokens to 2D and render a scatter-plot PNG,
    with point colors keyed by label.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = lookup._spaces[space]
    idx_map = {t: i for i, t in enumerate(lookup.vocab)}

    missing = [t for t in tokens if t not in idx_map]
    if missing:
        raise KeyError(f"tokens not in vocab: {missing!r}")

    indices = [idx_map[t] for t in tokens]
    vectors = s["sum_norm"][indices]
    coords = _compute_embedding(vectors, seed=seed)

    unique_labels = sorted(set(labels.get(t, "_other") for t in tokens))
    color_map = {lbl: plt.cm.tab10(i % 10) for i, lbl in enumerate(unique_labels)}
    colors = [color_map[labels.get(t, "_other")] for t in tokens]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=80, alpha=0.8)
    for i, t in enumerate(tokens):
        ax.annotate(t, (coords[i, 0], coords[i, 1]), fontsize=8, xytext=(3, 3),
                    textcoords="offset points")
    ax.set_title(title)
    ax.set_xlabel("axis 1")
    ax.set_ylabel("axis 2")

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[lbl],
                   markersize=10, label=lbl)
        for lbl in unique_labels
    ]
    ax.legend(handles=legend_handles, loc="best")

    if len(tokens) < UMAP_MIN_VOCAB:
        ax.text(0.02, 0.98, "(PCA fallback: vocab < 15)",
                transform=ax.transAxes, ha="left", va="top", fontsize=8, alpha=0.7)

    fig.tight_layout()
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
