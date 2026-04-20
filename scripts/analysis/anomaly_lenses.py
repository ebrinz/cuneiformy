"""
Civilization-agnostic anomaly lenses for aligned-embedding analysis.

Each lens takes numpy arrays + lookup maps + threshold parameters, and
returns a ranked list of anomaly rows. No I/O, no hardcoded paths, no
civilization-specific constants.

Used by scripts/analysis/anomaly_framework.py to build per-civilization atlases.

See: docs/superpowers/specs/2026-04-20-anomaly-atlas-design.md
"""
from __future__ import annotations

import numpy as np


def lens1_english_displacement(
    aligned_gemma: np.ndarray,
    source_vocab: list[str],
    target_gemma_vectors: np.ndarray,
    target_gemma_vocab_map: dict[str, int],
    anchors: list[dict],
    top_n: int,
    junk_target_glosses: frozenset[str],
    min_token_length: int,
    min_anchor_confidence: float,
) -> dict:
    """Lens 1: rank anchor pairs by cosine distance between aligned-source and
    target-native vectors. Low cosine similarity = translation that misses.

    Assumes `aligned_gemma` and `target_gemma_vectors` are pre-L2-normalized.

    Returns:
      {
        'rows_unfiltered': [top_n anchor rows sorted by ascending cosine],
        'rows_filtered': [top_n after junk-filter rules],
        'filter_rules_applied': [list of rule names]
      }
    """
    idx_map = {tok: i for i, tok in enumerate(source_vocab)}

    rows: list[dict] = []
    for anchor in anchors:
        src = anchor.get("sumerian", "").strip()
        tgt = anchor.get("english", "").lower().strip()
        if not src or not tgt:
            continue
        if src not in idx_map:
            continue
        if tgt not in target_gemma_vocab_map:
            continue
        source_vec = aligned_gemma[idx_map[src]]
        target_vec = target_gemma_vectors[target_gemma_vocab_map[tgt]]
        cos_sim = float(np.clip(np.dot(source_vec, target_vec), -1.0, 1.0))
        rows.append({
            "sumerian": src,
            "english": tgt,
            "cosine_similarity": cos_sim,
            "anchor_confidence": float(anchor.get("confidence", 0.0)),
            "source": anchor.get("source", ""),
        })

    rows.sort(key=lambda r: (r["cosine_similarity"], r["sumerian"]))

    def _passes_filter(row: dict) -> bool:
        if len(row["english"]) <= 2:
            return False
        if row["english"] in junk_target_glosses:
            return False
        if row["english"].isdigit():
            return False
        if len(row["sumerian"]) < min_token_length:
            return False
        if row["anchor_confidence"] < min_anchor_confidence:
            return False
        return True

    rows_filtered = [r for r in rows if _passes_filter(r)]

    return {
        "rows_unfiltered": rows[:top_n],
        "rows_filtered": rows_filtered[:top_n],
        "filter_rules_applied": [
            f"english_len>2", f"english_not_numeric", f"english_not_in_junk_set",
            f"sumerian_len>={min_token_length}",
            f"anchor_confidence>={min_anchor_confidence}",
        ],
    }


def lens3_isolation(
    aligned: np.ndarray,
    source_vocab: list[str],
    isolation_k: int,
    top_n: int,
    chunk_size: int = 500,
) -> dict:
    """Lens 3: pure within-source isolation. For each token, compute cosine
    distance to its k-th nearest neighbor. Rank descending (largest = most
    isolated).

    Assumes `aligned` is pre-L2-normalized.

    Returns:
      {
        'rows': [top_n rows sorted by descending distance-to-kth-neighbor],
        'histogram': {'bin_edges': [...], 'counts': [...]}  # over all tokens
      }
    """
    n = aligned.shape[0]
    distances = np.empty(n, dtype=np.float32)

    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        chunk = aligned[start:end]            # (chunk_size, dim)
        sims = chunk @ aligned.T              # (chunk_size, n)
        # Set self-similarity to -inf so it doesn't count.
        for i in range(chunk.shape[0]):
            sims[i, start + i] = -np.inf
        # Partition: the k largest similarities per row -> kth largest has index k-1.
        # Use -sims so np.partition finds smallest distances at index isolation_k-1.
        dists = 1.0 - sims
        # For isolation = distance to k-th nearest = k-th smallest distance (k starts at 1).
        # np.partition's kth=isolation_k-1 places the k-th smallest in that position.
        partitioned = np.partition(dists, isolation_k - 1, axis=1)
        distances[start:end] = partitioned[:, isolation_k - 1]

    # Build top-N ranked rows with nearest-5 neighbors for each.
    order = np.argsort(-distances, kind="stable")  # descending distance
    top_rows = []
    for idx in order[:top_n]:
        # Recompute sims for this row to get its nearest 5 neighbors.
        sims_row = aligned[idx] @ aligned.T
        sims_row[idx] = -np.inf
        nearest = np.argsort(-sims_row)[:5]
        top_rows.append({
            "sumerian": source_vocab[int(idx)],
            "distance_to_kth_neighbor": float(distances[idx]),
            "nearest_5_neighbors": [
                {"sumerian": source_vocab[int(j)], "cosine_similarity": float(sims_row[j])}
                for j in nearest
            ],
        })

    bin_edges = np.linspace(0.0, 2.0, 21)  # 20 bins from 0 to 2 (cosine distance range)
    counts, _ = np.histogram(distances, bins=bin_edges)

    return {
        "rows": top_rows,
        "histogram": {
            "bin_edges": bin_edges.tolist(),
            "counts": counts.tolist(),
        },
    }


def lens2_no_counterpart(
    aligned_gemma: np.ndarray,
    source_vocab: list[str],
    anchor_source_tokens: frozenset[str],
    target_gemma_vectors: np.ndarray,
    target_gemma_vocab: list[str],
    corpus_frequency: dict[str, int],
    top_n: int,
    chunk_size: int = 500,
) -> dict:
    """Lens 2: rank non-anchor source tokens by
       corpus_frequency * (1 - top_1_target_cosine)

    "High-value but no English counterpart." Assumes both `aligned_gemma` and
    `target_gemma_vectors` are L2-normalized.
    """
    non_anchor_indices = [
        i for i, tok in enumerate(source_vocab)
        if tok not in anchor_source_tokens
    ]
    rows: list[dict] = []
    for start in range(0, len(non_anchor_indices), chunk_size):
        batch_idx = non_anchor_indices[start : start + chunk_size]
        source_chunk = aligned_gemma[batch_idx]          # (chunk, dim)
        sims = source_chunk @ target_gemma_vectors.T     # (chunk, N_target)
        top1_target_idx = np.argmax(sims, axis=1)
        top1_cos = sims[np.arange(len(batch_idx)), top1_target_idx]
        for row_i, global_i in enumerate(batch_idx):
            tok = source_vocab[global_i]
            freq = corpus_frequency.get(tok, 0)
            cos = float(np.clip(top1_cos[row_i], -1.0, 1.0))
            score = freq * (1.0 - cos)
            rows.append({
                "sumerian": tok,
                "corpus_frequency": freq,
                "top1_english": target_gemma_vocab[int(top1_target_idx[row_i])],
                "top1_cosine": cos,
                "score": float(score),
            })

    rows.sort(key=lambda r: (-r["score"], r["sumerian"]))
    return {"rows": rows[:top_n]}


def _top_k_neighbors(aligned: np.ndarray, idx: int, k: int) -> set[int]:
    sims = aligned[idx] @ aligned.T
    sims[idx] = -np.inf
    top = np.argsort(-sims)[:k]
    return {int(j) for j in top}


def lens4_cross_space_divergence(
    aligned_gemma: np.ndarray,
    aligned_glove: np.ndarray,
    source_vocab: list[str],
    anchor_source_tokens: frozenset[str],
    top_n: int,
    neighbors_k: int = 10,
) -> dict:
    """Lens 4: Jaccard distance between a source token's top-K neighbors in
    two different aligned spaces (gemma vs glove). High divergence = the two
    alignments disagree on the word's semantic neighborhood — either noise in
    one space, or a real facet visible only to one target.

    Assumes both spaces' vectors are L2-normalized and aligned row-index-wise
    with `source_vocab`.
    """
    n = aligned_gemma.shape[0]
    assert aligned_glove.shape[0] == n, "spaces must share source vocab row ordering"

    rows: list[dict] = []
    for i in range(n):
        gemma_top = _top_k_neighbors(aligned_gemma, i, neighbors_k)
        glove_top = _top_k_neighbors(aligned_glove, i, neighbors_k)
        union = gemma_top | glove_top
        inter = gemma_top & glove_top
        if not union:
            jaccard_distance = 0.0
        else:
            jaccard_distance = 1.0 - (len(inter) / len(union))
        rows.append({
            "sumerian": source_vocab[i],
            "jaccard_distance": float(jaccard_distance),
            "top_k_gemma": [source_vocab[j] for j in sorted(gemma_top)],
            "top_k_glove": [source_vocab[j] for j in sorted(glove_top)],
        })

    rows.sort(key=lambda r: (-r["jaccard_distance"], r["sumerian"]))
    rows_anchor_only = [r for r in rows if r["sumerian"] in anchor_source_tokens]

    return {
        "rows_unfiltered": rows[:top_n],
        "rows_anchor_only": rows_anchor_only[:top_n],
    }
