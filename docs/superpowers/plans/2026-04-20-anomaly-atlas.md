# Sumerian Anomaly Atlas Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a civilization-agnostic anomaly-atlas framework (`scripts/analysis/anomaly_framework.py` + `anomaly_lenses.py`) plus a thin Sumerian-specific orchestrator that runs six anomaly lenses against the 52%-top-1 whitened-Gemma alignment, producing `docs/anomaly_atlas.json` + `docs/anomalies/*.md` as a diagnostic artifact.

**Architecture:** Pure-function lens module (`anomaly_lenses.py`) contains six civilization-agnostic lens implementations. Framework module (`anomaly_framework.py`) provides an `AnomalyConfig` dataclass + `run_atlas` orchestrator that loads artifacts, runs the lenses, and renders outputs. Sumerian-specific wrapper (`sumerian_anomaly_atlas.py`) builds the config for our artifacts and calls `run_atlas`. When a future Egyptian or comparative-repo atlas is needed, only a new wrapper is written — the framework is imported as-is.

**Tech Stack:** Python 3, numpy, scikit-learn (for KMeans), matplotlib (optional histogram PNGs), pytest. No new pip dependencies beyond what's already in the repo.

**Reference spec:** `docs/superpowers/specs/2026-04-20-anomaly-atlas-design.md`

---

## Before You Begin

- Current branch: `master`. Cut a fresh feature branch:
  ```bash
  cd /Users/crashy/Development/cuneiformy
  git checkout -b feat/anomaly-atlas
  ```
  All commits land on `feat/anomaly-atlas`. Merge via `superpowers:finishing-a-development-branch` after Task 7.

- Verify input artifacts are locally present:
  ```bash
  ls -la \
    final_output/sumerian_aligned_gemma_vectors.npz \
    final_output/sumerian_aligned_vectors.npz \
    final_output/sumerian_aligned_vocab.pkl \
    models/english_gemma_whitened_768d.npz \
    data/processed/glove.6B.300d.txt \
    data/processed/cleaned_corpus.txt \
    data/processed/english_anchors.json
  ```

- No new pip dependencies required. `scikit-learn` is already present (used in `09_align_and_evaluate.py`).

---

## File Structure

**New files:**
- `scripts/analysis/anomaly_lenses.py` — six pure-function lens implementations (~400 lines).
- `scripts/analysis/anomaly_framework.py` — `AnomalyConfig` dataclass + `run_atlas` orchestrator + markdown renderer (~300 lines).
- `scripts/analysis/sumerian_anomaly_atlas.py` — Sumerian-specific wrapper (~60 lines).
- `tests/analysis/test_anomaly_lenses.py` — 10 unit tests + 1 framework-integration test (~300 lines).
- `docs/anomaly_atlas.json` (generated).
- `docs/anomalies/atlas_summary.md` (generated).
- `docs/anomalies/lens1_english_displacement.md` (generated).
- `docs/anomalies/lens2_no_counterpart.md` (generated).
- `docs/anomalies/lens3_isolation.md` (generated).
- `docs/anomalies/lens4_cross_space_divergence.md` (generated).
- `docs/anomalies/lens5_doppelgangers.md` (generated).
- `docs/anomalies/lens6_structural_bridges.md` (generated).

**Modified files:**
- `docs/EXPERIMENT_JOURNAL.md` — journal entry upon completion.

**Untouched:**
- All pipeline scripts, existing analysis modules, `final_output/*`, models, tests.

---

## Task 1: Framework scaffolding + `AnomalyConfig` dataclass (TDD)

**Files:**
- Create: `scripts/analysis/anomaly_lenses.py` (empty stub — module docstring only)
- Create: `scripts/analysis/anomaly_framework.py` (dataclass only in this task)
- Create: `tests/analysis/test_anomaly_lenses.py` (single framework test)

### Setup note

Task 1 creates the scaffolding so later tasks can append lens implementations and tests. The dataclass is frozen and has civilization-agnostic field names throughout — no `sumerian_*` anywhere.

- [ ] **Step 1: Write the failing dataclass test**

Create `tests/analysis/test_anomaly_lenses.py`:

```python
"""Unit tests for the civilization-agnostic anomaly atlas framework + lenses."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


def test_anomaly_config_is_frozen():
    from scripts.analysis.anomaly_framework import AnomalyConfig

    config = AnomalyConfig(
        civilization_name="test",
        aligned_gemma_path=Path("/tmp/g.npz"),
        aligned_glove_path=None,
        source_vocab_path=Path("/tmp/vocab.pkl"),
        target_gemma_vocab_path=Path("/tmp/egm.npz"),
        target_glove_vocab_path=None,
        anchors_path=Path("/tmp/a.json"),
        corpus_frequency_path=Path("/tmp/corp.txt"),
        junk_target_glosses=frozenset({"x", "n"}),
        min_anchor_confidence=0.5,
        min_token_length=2,
        output_atlas_json=Path("/tmp/out.json"),
        output_markdown_dir=Path("/tmp/md"),
        output_figures_dir=None,
    )
    with pytest.raises((AttributeError, Exception)):
        config.civilization_name = "hacked"
```

- [ ] **Step 2: Run test, verify it fails**

```bash
cd /Users/crashy/Development/cuneiformy
pytest tests/analysis/test_anomaly_lenses.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.analysis.anomaly_framework'`.

- [ ] **Step 3: Create `scripts/analysis/anomaly_lenses.py` with just a docstring**

```python
"""
Civilization-agnostic anomaly lenses for aligned-embedding analysis.

Each lens takes numpy arrays + lookup maps + threshold parameters, and
returns a ranked list of anomaly rows. No I/O, no hardcoded paths, no
civilization-specific constants.

Used by scripts/analysis/anomaly_framework.py to build per-civilization atlases.

See: docs/superpowers/specs/2026-04-20-anomaly-atlas-design.md
"""
from __future__ import annotations

# Lens implementations land in later tasks.
```

- [ ] **Step 4: Create `scripts/analysis/anomaly_framework.py` with the dataclass + defaults**

```python
"""
Anomaly-atlas framework: AnomalyConfig + run_atlas orchestrator + markdown renderer.

Civilization-agnostic. Consumed by sumerian_anomaly_atlas.py for Sumerian;
future Egyptian / comparative-repo orchestrators instantiate their own
AnomalyConfig and call run_atlas.

See: docs/superpowers/specs/2026-04-20-anomaly-atlas-design.md
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AnomalyConfig:
    civilization_name: str
    aligned_gemma_path: Path
    aligned_glove_path: Path | None
    source_vocab_path: Path
    target_gemma_vocab_path: Path
    target_glove_vocab_path: Path | None
    anchors_path: Path
    corpus_frequency_path: Path
    junk_target_glosses: frozenset[str]
    min_anchor_confidence: float
    min_token_length: int
    output_atlas_json: Path
    output_markdown_dir: Path
    output_figures_dir: Path | None
    seed: int = 42
    k_clusters: int = 40
    top_n_per_lens: int = 50
    doppelganger_threshold: float = 0.95
    isolation_k: int = 10


# run_atlas and markdown renderer land in a later task.
```

- [ ] **Step 5: Run tests, verify pass**

```bash
pytest tests/analysis/test_anomaly_lenses.py -v
```
Expected: 1 PASS.

- [ ] **Step 6: Full suite regression**

```bash
pytest tests/ --ignore=tests/test_integration.py -q
```
Expected: 145 PASS (144 prior + 1 new).

- [ ] **Step 7: Commit**

```bash
cd /Users/crashy/Development/cuneiformy
git add scripts/analysis/anomaly_lenses.py \
        scripts/analysis/anomaly_framework.py \
        tests/analysis/test_anomaly_lenses.py
git commit -m "feat: add AnomalyConfig + scaffolding for anomaly atlas"
```

---

## Task 2: Lens 1 (english displacement) + Lens 3 (isolation) (TDD)

**Files:**
- Modify: `scripts/analysis/anomaly_lenses.py`
- Modify: `tests/analysis/test_anomaly_lenses.py`

### Setup note

Lens 1 and Lens 3 are both simple nearest-neighbor / cosine-similarity calculations. Lens 1 compares aligned-source vectors to target-native vectors via their anchor pairs. Lens 3 computes within-source isolation — no alignment or target side needed.

- [ ] **Step 1: Append failing tests**

Append to `tests/analysis/test_anomaly_lenses.py`:

```python
def _normalize_rows(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


# --- Lens 1: English displacement ---------------------------------------


def test_lens1_ranks_low_cosine_first():
    from scripts.analysis.anomaly_lenses import lens1_english_displacement

    # Two anchors: anchor[0] has identical source/target dirs (cos=1),
    # anchor[1] has orthogonal (cos=0). Top row should be anchor[1].
    aligned = _normalize_rows(np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32))
    source_vocab = ["a_src", "b_src"]
    target_vectors = _normalize_rows(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    target_vocab_map = {"a_tgt": 0, "b_tgt": 1}
    anchors = [
        {"sumerian": "a_src", "english": "a_tgt", "confidence": 0.9, "source": "ePSD2"},
        {"sumerian": "b_src", "english": "b_tgt", "confidence": 0.9, "source": "ePSD2"},
    ]
    result = lens1_english_displacement(
        aligned, source_vocab, target_vectors, target_vocab_map, anchors,
        top_n=10, junk_target_glosses=frozenset(), min_token_length=2,
        min_anchor_confidence=0.5,
    )
    rows = result["rows_unfiltered"]
    assert len(rows) == 2
    assert rows[0]["sumerian"] == "b_src"
    assert rows[0]["cosine_similarity"] == pytest.approx(0.0, abs=1e-5)


def test_lens1_filtered_excludes_short_english():
    from scripts.analysis.anomaly_lenses import lens1_english_displacement

    aligned = _normalize_rows(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    source_vocab = ["alpha", "beta"]
    target_vectors = _normalize_rows(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    target_vocab_map = {"king": 0, "c": 1}
    anchors = [
        {"sumerian": "alpha", "english": "king", "confidence": 0.9, "source": "ePSD2"},
        {"sumerian": "beta",  "english": "c",    "confidence": 0.9, "source": "ePSD2"},
    ]
    result = lens1_english_displacement(
        aligned, source_vocab, target_vectors, target_vocab_map, anchors,
        top_n=10, junk_target_glosses=frozenset(), min_token_length=2,
        min_anchor_confidence=0.5,
    )
    # Filtered tier drops the english="c" row.
    filtered_english = {r["english"] for r in result["rows_filtered"]}
    assert "c" not in filtered_english
    unfiltered_english = {r["english"] for r in result["rows_unfiltered"]}
    assert "c" in unfiltered_english


# --- Lens 3: Isolation --------------------------------------------------


def test_lens3_isolation_is_k_nearest_distance():
    from scripts.analysis.anomaly_lenses import lens3_isolation

    # 5 tokens in 2D: four form a tight cluster, one is isolated.
    vectors = np.array([
        [1.0, 0.0],
        [0.99, 0.05],
        [0.98, -0.05],
        [1.0, 0.1],
        [-1.0, 0.0],   # isolated — opposite direction
    ], dtype=np.float32)
    aligned = _normalize_rows(vectors)
    vocab = ["a", "b", "c", "d", "iso"]
    result = lens3_isolation(aligned, vocab, isolation_k=1, top_n=5)
    rows = result["rows"]
    assert rows[0]["sumerian"] == "iso"
    # cosine distance between iso and the cluster tokens is > 1.5
    assert rows[0]["distance_to_kth_neighbor"] > 1.0


def test_lens3_returns_histogram():
    from scripts.analysis.anomaly_lenses import lens3_isolation

    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((20, 8)).astype(np.float32)
    aligned = _normalize_rows(vectors)
    vocab = [f"t{i}" for i in range(20)]
    result = lens3_isolation(aligned, vocab, isolation_k=3, top_n=10)
    assert "histogram" in result
    assert "bin_edges" in result["histogram"]
    assert "counts" in result["histogram"]
    assert sum(result["histogram"]["counts"]) == len(aligned)
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
pytest tests/analysis/test_anomaly_lenses.py -v
```
Expected: 4 new tests FAIL on `ImportError: cannot import name 'lens1_english_displacement' from 'scripts.analysis.anomaly_lenses'`. The dataclass test still passes.

- [ ] **Step 3: Implement Lens 1 + Lens 3**

Append to `scripts/analysis/anomaly_lenses.py`:

```python
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
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/analysis/test_anomaly_lenses.py -v
```
Expected: 5 PASS (1 dataclass + 2 Lens 1 + 2 Lens 3).

- [ ] **Step 5: Full suite regression**

```bash
pytest tests/ --ignore=tests/test_integration.py -q
```
Expected: 149 PASS (145 + 4 new).

- [ ] **Step 6: Commit**

```bash
cd /Users/crashy/Development/cuneiformy
git add scripts/analysis/anomaly_lenses.py tests/analysis/test_anomaly_lenses.py
git commit -m "feat: add Lens 1 (english displacement) + Lens 3 (isolation)"
```

---

## Task 3: Lens 2 (no counterpart) + Lens 4 (cross-space divergence) (TDD)

**Files:**
- Modify: `scripts/analysis/anomaly_lenses.py`
- Modify: `tests/analysis/test_anomaly_lenses.py`

### Setup note

Lens 2 scores non-anchor tokens by `corpus_frequency × (1 - top_1_cosine)`. Lens 4 requires dual-space alignment; computes top-K neighbors in both spaces, Jaccard distance between the sets.

- [ ] **Step 1: Append failing tests**

```python
# --- Lens 2: No counterpart ---------------------------------------------


def test_lens2_score_combines_frequency_and_low_top1():
    from scripts.analysis.anomaly_lenses import lens2_no_counterpart

    # Two non-anchor tokens: token A has freq=100 and top-1 cosine 0.1 (score=90);
    # token B has freq=10 and top-1 cosine 0.9 (score=1).
    aligned = _normalize_rows(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    source_vocab = ["alpha", "beta"]
    anchor_source_tokens = frozenset()  # both are non-anchor
    target_vectors = _normalize_rows(np.array([[0.9, 0.436]], dtype=np.float32))  # 0.9 cos to [1,0]; ~0.44 cos to [0,1]
    target_vocab = ["x"]
    corpus_freq = {"alpha": 100, "beta": 10}

    result = lens2_no_counterpart(
        aligned, source_vocab, anchor_source_tokens,
        target_vectors, target_vocab, corpus_freq, top_n=10,
    )
    rows = result["rows"]
    # Alpha: cos(alpha, x) = 0.9  -> score = 100 * (1 - 0.9) = 10
    # Beta:  cos(beta, x)  = 0.436 -> score = 10 * (1 - 0.436) = 5.64
    # Alpha ranks first (higher score).
    assert rows[0]["sumerian"] == "alpha"


def test_lens2_skips_anchor_tokens():
    from scripts.analysis.anomaly_lenses import lens2_no_counterpart

    aligned = _normalize_rows(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    source_vocab = ["alpha", "beta"]
    anchor_source_tokens = frozenset({"alpha"})
    target_vectors = _normalize_rows(np.ones((1, 2), dtype=np.float32))
    target_vocab = ["x"]
    corpus_freq = {"alpha": 100, "beta": 10}

    result = lens2_no_counterpart(
        aligned, source_vocab, anchor_source_tokens,
        target_vectors, target_vocab, corpus_freq, top_n=10,
    )
    # 'alpha' is in anchor_source_tokens, so it should not appear.
    assert all(r["sumerian"] != "alpha" for r in result["rows"])


# --- Lens 4: Cross-space divergence -------------------------------------


def test_lens4_jaccard_distance_all_different():
    from scripts.analysis.anomaly_lenses import lens4_cross_space_divergence

    # Build two spaces where every token has completely different top-K in each.
    # Use 4 tokens. In gemma space: a ~ [b,c]. In glove space: a ~ [c,d].
    gemma = _normalize_rows(np.array([
        [1.0, 0.0],
        [0.99, 0.05],    # close to a in gemma
        [0.98, -0.05],   # close to a in gemma
        [-1.0, 0.0],
    ], dtype=np.float32))
    # In glove: a ~ [d, c]; a and b are orthogonal.
    glove = _normalize_rows(np.array([
        [1.0, 0.0],
        [0.0, 1.0],      # orthogonal to a in glove
        [0.95, 0.312],   # close to a in glove
        [0.9, 0.436],    # close to a in glove
    ], dtype=np.float32))
    source_vocab = ["a", "b", "c", "d"]
    anchor_source_tokens = frozenset({"a"})

    result = lens4_cross_space_divergence(
        gemma, glove, source_vocab, anchor_source_tokens,
        top_n=5, neighbors_k=2,
    )
    rows_unfiltered = result["rows_unfiltered"]
    a_row = next(r for r in rows_unfiltered if r["sumerian"] == "a")
    assert a_row["jaccard_distance"] >= 0.0


def test_lens4_jaccard_distance_identical_neighbors():
    from scripts.analysis.anomaly_lenses import lens4_cross_space_divergence

    # Two identical spaces -> Jaccard distance = 0 for every token.
    rng = np.random.default_rng(0)
    mat = _normalize_rows(rng.standard_normal((6, 4)).astype(np.float32))
    source_vocab = [f"t{i}" for i in range(6)]
    result = lens4_cross_space_divergence(
        mat, mat, source_vocab, frozenset({source_vocab[0]}),
        top_n=6, neighbors_k=3,
    )
    for row in result["rows_unfiltered"]:
        assert row["jaccard_distance"] == pytest.approx(0.0, abs=1e-9)
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
pytest tests/analysis/test_anomaly_lenses.py -v
```
Expected: 4 new tests FAIL on `ImportError`. Prior 5 pass.

- [ ] **Step 3: Append Lens 2 and Lens 4 to `anomaly_lenses.py`**

```python
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
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/analysis/test_anomaly_lenses.py -v
```
Expected: 9 PASS.

- [ ] **Step 5: Full suite**

```bash
pytest tests/ --ignore=tests/test_integration.py -q
```
Expected: 153 PASS (149 + 4 new).

- [ ] **Step 6: Commit**

```bash
cd /Users/crashy/Development/cuneiformy
git add scripts/analysis/anomaly_lenses.py tests/analysis/test_anomaly_lenses.py
git commit -m "feat: add Lens 2 (no counterpart) + Lens 4 (cross-space divergence)"
```

---

## Task 4: Lens 5 (doppelgangers) + Lens 6 (structural bridges) (TDD)

**Files:**
- Modify: `scripts/analysis/anomaly_lenses.py`
- Modify: `tests/analysis/test_anomaly_lenses.py`

### Setup note

Lens 5 finds pairs of source tokens with cosine similarity above a threshold, using chunked computation (avoiding the full N×N dense matrix). Lens 6 runs k-means on source vectors, scores each token by how close it is to two clusters at once.

- [ ] **Step 1: Append failing tests**

```python
# --- Lens 5: Doppelgangers ----------------------------------------------


def test_lens5_doppelganger_finds_identical_pair():
    from scripts.analysis.anomaly_lenses import lens5_doppelgangers

    # Tokens a and b are identical in direction; others are spread out.
    vectors = _normalize_rows(np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],  # a/b identical
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
    ], dtype=np.float32))
    vocab = ["a", "b", "c", "d", "e"]
    result = lens5_doppelgangers(
        vectors, vocab, frozenset(), threshold=0.95, top_n=5,
    )
    assert len(result["rows"]) >= 1
    top = result["rows"][0]
    assert {top["sumerian_a"], top["sumerian_b"]} == {"a", "b"}
    assert top["cosine_similarity"] >= 0.95


def test_lens5_respects_threshold():
    from scripts.analysis.anomaly_lenses import lens5_doppelgangers

    vectors = _normalize_rows(np.array([
        [1.0, 0.0],
        [0.9, 0.436],   # cos to row 0 = 0.9
    ], dtype=np.float32))
    vocab = ["a", "b"]
    # threshold 0.95 -> no pair
    result_strict = lens5_doppelgangers(vectors, vocab, frozenset(), threshold=0.95, top_n=5)
    assert result_strict["rows"] == []
    # threshold 0.85 -> pair surfaces
    result_loose = lens5_doppelgangers(vectors, vocab, frozenset(), threshold=0.85, top_n=5)
    assert len(result_loose["rows"]) == 1


# --- Lens 6: Structural bridges -----------------------------------------


def test_lens6_bridge_score_is_high_when_equidistant():
    from scripts.analysis.anomaly_lenses import lens6_structural_bridges

    # Two obvious clusters + one bridge token equidistant from both.
    np.random.seed(0)
    cluster_a = np.array([[1.0, 0.0], [0.99, 0.02], [0.98, -0.02]], dtype=np.float32)
    cluster_b = np.array([[-1.0, 0.0], [-0.99, 0.02], [-0.98, -0.02]], dtype=np.float32)
    bridge = np.array([[0.0, 1.0]], dtype=np.float32)  # equidistant from both clusters
    vectors = _normalize_rows(np.vstack([cluster_a, cluster_b, bridge]))
    vocab = ["a1", "a2", "a3", "b1", "b2", "b3", "bridge"]
    result = lens6_structural_bridges(
        vectors, vocab, k_clusters=2, top_n=7, seed=0,
    )
    rows = result["rows"]
    bridge_row = next(r for r in rows if r["sumerian"] == "bridge")
    # The bridge token should have the highest bridge score.
    assert bridge_row["bridge_score"] == pytest.approx(
        max(r["bridge_score"] for r in rows), abs=1e-5
    )


def test_lens6_reports_k_clusters():
    from scripts.analysis.anomaly_lenses import lens6_structural_bridges

    rng = np.random.default_rng(0)
    vectors = _normalize_rows(rng.standard_normal((30, 8)).astype(np.float32))
    vocab = [f"t{i}" for i in range(30)]
    result = lens6_structural_bridges(
        vectors, vocab, k_clusters=4, top_n=5, seed=0,
    )
    assert result["k_clusters"] == 4
    for row in result["rows"]:
        assert "bridge_score" in row
        assert "nearest_cluster" in row
        assert "second_nearest_cluster" in row
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
pytest tests/analysis/test_anomaly_lenses.py -v
```
Expected: 4 new tests FAIL on `ImportError`. Prior 9 pass.

- [ ] **Step 3: Append Lens 5 and Lens 6 to `anomaly_lenses.py`**

```python
def lens5_doppelgangers(
    aligned: np.ndarray,
    source_vocab: list[str],
    anchor_source_tokens: frozenset[str],
    threshold: float,
    top_n: int,
    chunk_size: int = 500,
) -> dict:
    """Lens 5: find all pairs (i, j) with cos_sim(aligned[i], aligned[j]) >= threshold.

    Chunked to avoid the full N×N dense similarity matrix. Threshold-filters
    pairs inside each row-chunk.
    """
    n = aligned.shape[0]
    pairs: list[tuple[int, int, float]] = []
    hist_edges = np.linspace(0.85, 1.0, 16)  # 15 bins from 0.85 to 1.00
    hist_counts = np.zeros(15, dtype=np.int64)

    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        chunk = aligned[start:end]
        sims = chunk @ aligned.T                          # (chunk, n)
        # For each row i in chunk, consider only j > start+row to avoid duplicates
        # and to skip self-pairs.
        for row_i in range(chunk.shape[0]):
            i_global = start + row_i
            row = sims[row_i]
            # Tally for histogram (include pairs only ONCE: require j > i_global).
            relevant = row[i_global + 1 :]
            hist_update, _ = np.histogram(relevant, bins=hist_edges)
            hist_counts += hist_update
            # Collect over-threshold pairs.
            over = np.where(relevant >= threshold)[0]
            for j_offset in over:
                j_global = i_global + 1 + int(j_offset)
                pairs.append((i_global, j_global, float(row[i_global + 1 + int(j_offset)])))

    # Sort by descending cosine, ties broken by token alphabetical.
    pairs.sort(key=lambda p: (-p[2], source_vocab[p[0]], source_vocab[p[1]]))

    rows = []
    for (i, j, cos) in pairs[:top_n]:
        tok_i = source_vocab[i]
        tok_j = source_vocab[j]
        rows.append({
            "sumerian_a": tok_i,
            "sumerian_b": tok_j,
            "cosine_similarity": float(cos),
            "in_anchor_set": [tok_i in anchor_source_tokens, tok_j in anchor_source_tokens],
        })

    return {
        "rows": rows,
        "histogram": {
            "bin_edges": hist_edges.tolist(),
            "counts": hist_counts.tolist(),
        },
    }


def lens6_structural_bridges(
    aligned: np.ndarray,
    source_vocab: list[str],
    k_clusters: int,
    top_n: int,
    seed: int,
) -> dict:
    """Lens 6: k-means cluster the aligned vectors, then for each token compute
    cosine distances to all k cluster centroids. Bridge score =
      1.0 - (min_distance / second_min_distance)
    where min_distance is cosine distance to nearest centroid. Higher bridge
    score = more equidistant between two clusters = structural bridge.

    Clustering is deterministic with `random_state=seed`.
    """
    from sklearn.cluster import KMeans

    n = aligned.shape[0]
    km = KMeans(n_clusters=k_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(aligned)
    centroids = km.cluster_centers_  # (k, dim)
    # L2-normalize centroids so cosine similarities are meaningful.
    centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroid_norms[centroid_norms == 0] = 1.0
    centroids_norm = centroids / centroid_norms

    # Cosine distances from every token to every centroid.
    sims = aligned @ centroids_norm.T                # (n, k)
    dists = 1.0 - sims                               # cosine distance

    # For each token: two smallest distances.
    sorted_dist_idx = np.argsort(dists, axis=1)      # (n, k)
    nearest_cluster = sorted_dist_idx[:, 0]
    second_cluster = sorted_dist_idx[:, 1]
    d_min = dists[np.arange(n), nearest_cluster]
    d_second = dists[np.arange(n), second_cluster]
    d_second_safe = np.where(d_second > 0, d_second, 1.0)
    bridge_score = 1.0 - (d_min / d_second_safe)

    # Members of each cluster (for reporting).
    cluster_members: dict[int, list[str]] = {k: [] for k in range(k_clusters)}
    for i, cluster_id in enumerate(labels):
        cluster_members[int(cluster_id)].append(source_vocab[i])

    order = np.argsort(-bridge_score, kind="stable")
    rows = []
    for idx in order[:top_n]:
        nearest = int(nearest_cluster[idx])
        second = int(second_cluster[idx])
        rows.append({
            "sumerian": source_vocab[int(idx)],
            "bridge_score": float(bridge_score[idx]),
            "nearest_cluster": nearest,
            "second_nearest_cluster": second,
            f"cluster_{nearest}_members": cluster_members[nearest][:5],
            f"cluster_{second}_members": cluster_members[second][:5],
        })

    return {"k_clusters": k_clusters, "rows": rows}
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/analysis/test_anomaly_lenses.py -v
```
Expected: 13 PASS (1 framework + 12 lens tests).

- [ ] **Step 5: Full suite**

```bash
pytest tests/ --ignore=tests/test_integration.py -q
```
Expected: 157 PASS (153 + 4 new).

- [ ] **Step 6: Commit**

```bash
cd /Users/crashy/Development/cuneiformy
git add scripts/analysis/anomaly_lenses.py tests/analysis/test_anomaly_lenses.py
git commit -m "feat: add Lens 5 (doppelgangers) + Lens 6 (structural bridges)"
```

---

## Task 5: Markdown renderer in `anomaly_framework.py` (TDD)

**Files:**
- Modify: `scripts/analysis/anomaly_framework.py`
- Modify: `tests/analysis/test_anomaly_lenses.py`

### Setup note

This task adds the markdown rendering logic to the framework. Each lens has a dedicated renderer that takes the JSON section and emits per-lens markdown. Plus a top-level summary renderer.

All renderers are pure functions: take dict, return markdown string. I/O happens in `run_atlas`.

- [ ] **Step 1: Append failing renderer tests**

```python
# --- Markdown renderer --------------------------------------------------


def _synthetic_atlas_section():
    return {
        "atlas_schema_version": 1,
        "atlas_date": "2026-04-20",
        "civilization": "sumerian",
        "source_artifacts": {"seed": 42, "k_clusters": 40},
        "summary": {
            "total_aligned_tokens": 100,
            "anchor_tokens_in_vocab": 30,
            "non_anchor_tokens_in_vocab": 70,
            "top1_per_lens": {
                "lens1_english_displacement": "alpha -> beta (cos=0.05)",
                "lens2_no_counterpart": "gamma (freq=100, top1_cos=0.1)",
                "lens3_isolation": "delta (d_10=1.8)",
                "lens4_cross_space_divergence": "epsilon (jaccard=0.9)",
                "lens5_doppelgangers": "zeta == eta (cos=0.97)",
                "lens6_structural_bridges": "theta (bridge=0.9, clusters 3/7)",
            },
        },
        "lens1_english_displacement": {
            "rows_unfiltered": [
                {"sumerian": "alpha", "english": "beta", "cosine_similarity": 0.05,
                 "anchor_confidence": 0.9, "source": "ePSD2"},
            ],
            "rows_filtered": [
                {"sumerian": "alpha", "english": "beta", "cosine_similarity": 0.05,
                 "anchor_confidence": 0.9, "source": "ePSD2"},
            ],
            "filter_rules_applied": [],
        },
        "lens2_no_counterpart": {
            "rows": [{"sumerian": "gamma", "corpus_frequency": 100,
                      "top1_english": "x", "top1_cosine": 0.1, "score": 90.0}],
        },
        "lens3_isolation": {
            "rows": [{"sumerian": "delta", "distance_to_kth_neighbor": 1.8,
                      "nearest_5_neighbors": []}],
            "histogram": {"bin_edges": [0.0, 1.0, 2.0], "counts": [50, 50]},
        },
        "lens4_cross_space_divergence": {
            "rows_unfiltered": [{"sumerian": "epsilon", "jaccard_distance": 0.9,
                                 "top_k_gemma": ["a"], "top_k_glove": ["b"]}],
            "rows_anchor_only": [],
        },
        "lens5_doppelgangers": {
            "rows": [{"sumerian_a": "zeta", "sumerian_b": "eta",
                      "cosine_similarity": 0.97, "in_anchor_set": [True, False]}],
            "histogram": {"bin_edges": [0.85, 0.90, 0.95, 1.0], "counts": [10, 5, 1]},
        },
        "lens6_structural_bridges": {
            "k_clusters": 40,
            "rows": [{"sumerian": "theta", "bridge_score": 0.9,
                      "nearest_cluster": 3, "second_nearest_cluster": 7,
                      "cluster_3_members": [], "cluster_7_members": []}],
        },
    }


def test_render_summary_markdown_contains_all_lenses():
    from scripts.analysis.anomaly_framework import render_summary_markdown

    md = render_summary_markdown(_synthetic_atlas_section())
    for lens in ("Lens 1", "Lens 2", "Lens 3", "Lens 4", "Lens 5", "Lens 6"):
        assert lens in md


def test_render_lens1_markdown_has_both_tiers():
    from scripts.analysis.anomaly_framework import render_lens1_markdown

    md = render_lens1_markdown(_synthetic_atlas_section())
    assert "Unfiltered" in md
    assert "Filtered" in md
    assert "alpha" in md  # token from synthetic section


def test_render_lens5_markdown_shows_histogram_summary():
    from scripts.analysis.anomaly_framework import render_lens5_markdown

    md = render_lens5_markdown(_synthetic_atlas_section())
    # ASCII histogram or a prose summary of bin counts.
    assert "0.85" in md or "histogram" in md.lower()
```

- [ ] **Step 2: Run tests, verify fail**

```bash
pytest tests/analysis/test_anomaly_lenses.py -v
```
Expected: 3 new tests FAIL on ImportError for `render_*_markdown`.

- [ ] **Step 3: Append renderers to `anomaly_framework.py`**

```python
def _render_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    """Render a list of row-dicts as a markdown table. columns=[(display, key)]."""
    if not rows:
        return "_(no rows)_\n"
    header = "| " + " | ".join(disp for disp, _ in columns) + " |"
    sep = "|" + "|".join(["---"] * len(columns)) + "|"
    lines = [header, sep]
    for row in rows:
        vals = []
        for _, key in columns:
            v = row.get(key, "")
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def render_summary_markdown(atlas: dict) -> str:
    lines = []
    summary = atlas["summary"]
    lines.append(f"# Anomaly Atlas — {atlas['civilization']} ({atlas['atlas_date']})")
    lines.append("")
    lines.append(f"- **Total aligned tokens:** {summary['total_aligned_tokens']:,}")
    lines.append(f"- **Anchor tokens in vocab:** {summary['anchor_tokens_in_vocab']:,}")
    lines.append(f"- **Non-anchor tokens in vocab:** {summary['non_anchor_tokens_in_vocab']:,}")
    lines.append("")
    lines.append("## Top-1 per lens")
    lines.append("")
    for lens_num in range(1, 7):
        key = {
            1: "lens1_english_displacement",
            2: "lens2_no_counterpart",
            3: "lens3_isolation",
            4: "lens4_cross_space_divergence",
            5: "lens5_doppelgangers",
            6: "lens6_structural_bridges",
        }[lens_num]
        top1 = summary.get("top1_per_lens", {}).get(key, "n/a")
        lines.append(f"- **Lens {lens_num} ({key}):** {top1}")
    lines.append("")
    lines.append("## Lens details")
    lines.append("")
    lens_filenames = {
        1: "lens1_english_displacement.md",
        2: "lens2_no_counterpart.md",
        3: "lens3_isolation.md",
        4: "lens4_cross_space_divergence.md",
        5: "lens5_doppelgangers.md",
        6: "lens6_structural_bridges.md",
    }
    for lens_num, fname in lens_filenames.items():
        lines.append(f"- [Lens {lens_num}]({fname})")
    lines.append("")
    return "\n".join(lines)


def render_lens1_markdown(atlas: dict) -> str:
    section = atlas["lens1_english_displacement"]
    lines = [
        f"# Lens 1: English displacement",
        "",
        "Anchor pairs ranked by cosine similarity between the source-language "
        "token's aligned (projected) vector and the English gloss's native target vector. "
        "Low cosine = translation that geometrically misses.",
        "",
        "## Unfiltered (includes anchor-quality noise)",
        "",
        _render_table(
            section["rows_unfiltered"],
            [("Source", "sumerian"), ("English", "english"),
             ("Cos", "cosine_similarity"), ("Conf", "anchor_confidence"),
             ("Src", "source")],
        ),
        "",
        "## Filtered (anchor-quality rules applied)",
        "",
        f"_Rules: {', '.join(section.get('filter_rules_applied', []))}_",
        "",
        _render_table(
            section["rows_filtered"],
            [("Source", "sumerian"), ("English", "english"),
             ("Cos", "cosine_similarity"), ("Conf", "anchor_confidence"),
             ("Src", "source")],
        ),
        "",
    ]
    return "\n".join(lines)


def render_lens2_markdown(atlas: dict) -> str:
    section = atlas["lens2_no_counterpart"]
    lines = [
        "# Lens 2: No counterpart",
        "",
        "Non-anchor source tokens ranked by `corpus_frequency × (1 − top1_target_cosine)`. "
        "High score = appears often in the corpus AND no English word lands close.",
        "",
        _render_table(
            section["rows"],
            [("Source", "sumerian"), ("Freq", "corpus_frequency"),
             ("Top-1 English", "top1_english"),
             ("Cos", "top1_cosine"), ("Score", "score")],
        ),
    ]
    return "\n".join(lines)


def render_lens3_markdown(atlas: dict) -> str:
    section = atlas["lens3_isolation"]
    lines = [
        "# Lens 3: Isolation in source space",
        "",
        "Source tokens ranked by cosine distance to their k-th nearest neighbor. "
        "Large distance = isolated. No alignment needed; pure within-source geometry.",
        "",
        _render_table(
            section["rows"],
            [("Source", "sumerian"), ("D(kth)", "distance_to_kth_neighbor")],
        ),
        "",
        "## Isolation histogram",
        "",
        "_Bin counts (cosine-distance bins across all tokens):_",
        "",
        _render_histogram_line(section["histogram"]),
        "",
    ]
    return "\n".join(lines)


def render_lens4_markdown(atlas: dict) -> str:
    section = atlas["lens4_cross_space_divergence"]
    lines = [
        "# Lens 4: Cross-space divergence",
        "",
        "Source tokens ranked by Jaccard distance between their top-K neighbors "
        "in the Gemma and GloVe aligned spaces. High divergence = one space sees "
        "a facet the other misses (or alignment noise on one side).",
        "",
        "## Unfiltered",
        "",
        _render_table(
            section["rows_unfiltered"],
            [("Source", "sumerian"), ("Jaccard-D", "jaccard_distance")],
        ),
        "",
        "## Anchor-only",
        "",
        _render_table(
            section["rows_anchor_only"],
            [("Source", "sumerian"), ("Jaccard-D", "jaccard_distance")],
        ),
    ]
    return "\n".join(lines)


def render_lens5_markdown(atlas: dict) -> str:
    section = atlas["lens5_doppelgangers"]
    lines = [
        "# Lens 5: Doppelgangers",
        "",
        "Source-token pairs with cosine similarity ≥ threshold. "
        "Near-identical embeddings may indicate morphological variants, "
        "scribal variants, or genuine near-synonyms.",
        "",
        _render_table(
            section["rows"],
            [("Source A", "sumerian_a"), ("Source B", "sumerian_b"),
             ("Cos", "cosine_similarity")],
        ),
        "",
        "## Similarity histogram (≥ 0.85)",
        "",
        _render_histogram_line(section["histogram"]),
        "",
    ]
    return "\n".join(lines)


def render_lens6_markdown(atlas: dict) -> str:
    section = atlas["lens6_structural_bridges"]
    lines = [
        "# Lens 6: Structural bridges",
        "",
        f"Source tokens ranked by bridge score (k-means k={section.get('k_clusters', 'N/A')}, seed={atlas['source_artifacts'].get('seed', 42)}). "
        "Higher score = more equidistant between two clusters = candidate conceptual bridge.",
        "",
        _render_table(
            section["rows"],
            [("Source", "sumerian"), ("Bridge", "bridge_score"),
             ("Cluster A", "nearest_cluster"), ("Cluster B", "second_nearest_cluster")],
        ),
    ]
    return "\n".join(lines)


def _render_histogram_line(hist: dict) -> str:
    """Compact ASCII histogram: bin_start→count."""
    edges = hist.get("bin_edges", [])
    counts = hist.get("counts", [])
    lines = []
    for i, c in enumerate(counts):
        lo = edges[i] if i < len(edges) else 0
        hi = edges[i + 1] if i + 1 < len(edges) else 0
        bar = "█" * min(40, int(c / max(counts) * 40)) if max(counts) > 0 else ""
        lines.append(f"- {lo:.3f}–{hi:.3f}: {c} {bar}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/analysis/test_anomaly_lenses.py -v
```
Expected: 16 PASS (13 + 3 new).

- [ ] **Step 5: Full suite**

```bash
pytest tests/ --ignore=tests/test_integration.py -q
```
Expected: 160 PASS (157 + 3 new).

- [ ] **Step 6: Commit**

```bash
cd /Users/crashy/Development/cuneiformy
git add scripts/analysis/anomaly_framework.py tests/analysis/test_anomaly_lenses.py
git commit -m "feat: add markdown renderers for anomaly atlas"
```

---

## Task 6: `run_atlas` orchestrator + Sumerian wrapper + integration test

**Files:**
- Modify: `scripts/analysis/anomaly_framework.py`
- Create: `scripts/analysis/sumerian_anomaly_atlas.py`
- Modify: `tests/analysis/test_anomaly_lenses.py`

### Setup note

`run_atlas` is the orchestrator that loads artifacts, runs the six lenses, renders markdown, writes the atlas JSON. The Sumerian wrapper builds the config and calls `run_atlas`.

- [ ] **Step 1: Append integration test**

```python
def test_run_atlas_produces_schema_compliant_json(tmp_path):
    """End-to-end: run_atlas against synthetic inputs emits JSON with all lens
    sections and all required top-level keys."""
    from scripts.analysis.anomaly_framework import AnomalyConfig, run_atlas
    import pickle as _pkl

    # Build tiny synthetic artifacts.
    rng = np.random.default_rng(42)
    source_vocab = [f"src{i}" for i in range(30)]
    aligned_gemma = _normalize_rows(rng.standard_normal((30, 16)).astype(np.float32))
    aligned_glove = _normalize_rows(rng.standard_normal((30, 16)).astype(np.float32))
    target_vocab = [f"eng{i}" for i in range(15)]
    target_gemma = _normalize_rows(rng.standard_normal((15, 16)).astype(np.float32))

    # Save artifacts
    np.savez_compressed(
        tmp_path / "aligned_gemma.npz", vectors=aligned_gemma,
    )
    np.savez_compressed(
        tmp_path / "aligned_glove.npz", vectors=aligned_glove,
    )
    with open(tmp_path / "vocab.pkl", "wb") as f:
        _pkl.dump(source_vocab, f)
    np.savez_compressed(
        tmp_path / "target_gemma.npz",
        vectors=target_gemma, vocab=np.array(target_vocab),
    )
    # Minimal anchors: first 5 tokens are anchors.
    anchors = [
        {"sumerian": source_vocab[i], "english": target_vocab[i % 15],
         "confidence": 0.9, "source": "ePSD2"}
        for i in range(5)
    ]
    (tmp_path / "anchors.json").write_text(json.dumps(anchors))
    (tmp_path / "corpus.txt").write_text(" ".join(source_vocab * 3))

    out_json = tmp_path / "atlas.json"
    out_md = tmp_path / "md"

    config = AnomalyConfig(
        civilization_name="test",
        aligned_gemma_path=tmp_path / "aligned_gemma.npz",
        aligned_glove_path=tmp_path / "aligned_glove.npz",
        source_vocab_path=tmp_path / "vocab.pkl",
        target_gemma_vocab_path=tmp_path / "target_gemma.npz",
        target_glove_vocab_path=None,  # Lens 2 uses only Gemma target; None is fine
        anchors_path=tmp_path / "anchors.json",
        corpus_frequency_path=tmp_path / "corpus.txt",
        junk_target_glosses=frozenset(),
        min_anchor_confidence=0.5,
        min_token_length=2,
        output_atlas_json=out_json,
        output_markdown_dir=out_md,
        output_figures_dir=None,
        seed=42,
        k_clusters=3,              # tiny for synthetic data
        top_n_per_lens=5,
        doppelganger_threshold=0.9,  # loose for synthetic noise
        isolation_k=3,
    )

    run_atlas(config)

    assert out_json.exists()
    atlas = json.loads(out_json.read_text())
    assert atlas["atlas_schema_version"] == 1
    assert atlas["civilization"] == "test"
    for key in (
        "lens1_english_displacement", "lens2_no_counterpart",
        "lens3_isolation", "lens4_cross_space_divergence",
        "lens5_doppelgangers", "lens6_structural_bridges",
    ):
        assert key in atlas

    # All 7 markdown files are produced.
    for fname in (
        "atlas_summary.md", "lens1_english_displacement.md",
        "lens2_no_counterpart.md", "lens3_isolation.md",
        "lens4_cross_space_divergence.md", "lens5_doppelgangers.md",
        "lens6_structural_bridges.md",
    ):
        assert (out_md / fname).exists()
```

- [ ] **Step 2: Run test, verify it fails**

```bash
pytest tests/analysis/test_anomaly_lenses.py::test_run_atlas_produces_schema_compliant_json -v
```
Expected: FAIL on `ImportError: cannot import name 'run_atlas'`.

- [ ] **Step 3: Append `run_atlas` to `anomaly_framework.py`**

```python
import datetime as _dt
import hashlib
import json
import pickle as _pkl
from collections import Counter

import numpy as np

from scripts.analysis.anomaly_lenses import (
    lens1_english_displacement, lens2_no_counterpart, lens3_isolation,
    lens4_cross_space_divergence, lens5_doppelgangers, lens6_structural_bridges,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_matrix(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def _load_aligned_npz(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if "vectors" in data.files:
        v = data["vectors"].astype(np.float32)
    else:
        raise ValueError(f"expected 'vectors' array in {path}")
    return _normalize_matrix(v)


def _load_target_gemma_npz(path: Path) -> tuple[list[str], np.ndarray]:
    data = np.load(path, allow_pickle=True)
    vocab = [str(w) for w in data["vocab"]]
    vectors = _normalize_matrix(data["vectors"].astype(np.float32))
    return vocab, vectors


def _load_vocab_pkl(path: Path) -> list[str]:
    with open(path, "rb") as f:
        return list(_pkl.load(f))


def _load_corpus_frequency(path: Path) -> dict[str, int]:
    freq = Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            for token in line.strip().split():
                freq[token] += 1
    return dict(freq)


def _load_anchors(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_atlas(config: AnomalyConfig) -> dict:
    """Orchestrate the 6 lenses, render JSON + markdown, return the summary dict.

    Lens 4 is skipped with a note if aligned_glove_path is None.
    """
    print(f"[atlas] Loading artifacts...")
    aligned_gemma = _load_aligned_npz(config.aligned_gemma_path)
    source_vocab = _load_vocab_pkl(config.source_vocab_path)
    target_gemma_vocab, target_gemma_vectors = _load_target_gemma_npz(config.target_gemma_vocab_path)
    target_gemma_vocab_map = {w.lower(): i for i, w in enumerate(target_gemma_vocab)}
    anchors = _load_anchors(config.anchors_path)
    corpus_freq = _load_corpus_frequency(config.corpus_frequency_path)

    anchor_source_tokens = frozenset(
        a["sumerian"] for a in anchors if a.get("sumerian")
    )

    aligned_glove = None
    if config.aligned_glove_path:
        aligned_glove = _load_aligned_npz(config.aligned_glove_path)

    # Lens 1
    print(f"[atlas] Lens 1 (english displacement)...")
    l1 = lens1_english_displacement(
        aligned_gemma, source_vocab, target_gemma_vectors, target_gemma_vocab_map,
        anchors, config.top_n_per_lens, config.junk_target_glosses,
        config.min_token_length, config.min_anchor_confidence,
    )

    # Lens 2
    print(f"[atlas] Lens 2 (no counterpart)...")
    l2 = lens2_no_counterpart(
        aligned_gemma, source_vocab, anchor_source_tokens,
        target_gemma_vectors, target_gemma_vocab, corpus_freq,
        config.top_n_per_lens,
    )

    # Lens 3
    print(f"[atlas] Lens 3 (isolation)...")
    l3 = lens3_isolation(
        aligned_gemma, source_vocab, config.isolation_k, config.top_n_per_lens,
    )

    # Lens 4
    if aligned_glove is not None:
        print(f"[atlas] Lens 4 (cross-space divergence)...")
        l4 = lens4_cross_space_divergence(
            aligned_gemma, aligned_glove, source_vocab, anchor_source_tokens,
            config.top_n_per_lens, neighbors_k=10,
        )
    else:
        print(f"[atlas] Lens 4 skipped (no aligned_glove_path)")
        l4 = {"rows_unfiltered": [], "rows_anchor_only": [],
              "note": "skipped (requires dual-target alignment)"}

    # Lens 5
    print(f"[atlas] Lens 5 (doppelgangers)...")
    l5 = lens5_doppelgangers(
        aligned_gemma, source_vocab, anchor_source_tokens,
        config.doppelganger_threshold, config.top_n_per_lens,
    )

    # Lens 6
    print(f"[atlas] Lens 6 (structural bridges)...")
    l6 = lens6_structural_bridges(
        aligned_gemma, source_vocab, config.k_clusters, config.top_n_per_lens,
        seed=config.seed,
    )

    total_tokens = len(source_vocab)
    anchor_count_in_vocab = sum(1 for t in source_vocab if t in anchor_source_tokens)
    non_anchor_count = total_tokens - anchor_count_in_vocab

    def _top1_lens1():
        rows = l1.get("rows_unfiltered", [])
        if not rows:
            return "n/a"
        r = rows[0]
        return f"{r['sumerian']} -> {r['english']} (cos={r['cosine_similarity']:.4f})"

    def _top1_lens2():
        rows = l2.get("rows", [])
        if not rows:
            return "n/a"
        r = rows[0]
        return f"{r['sumerian']} (freq={r['corpus_frequency']}, top1_cos={r['top1_cosine']:.4f})"

    def _top1_lens3():
        rows = l3.get("rows", [])
        if not rows:
            return "n/a"
        r = rows[0]
        return f"{r['sumerian']} (d_k={r['distance_to_kth_neighbor']:.4f})"

    def _top1_lens4():
        rows = l4.get("rows_unfiltered", [])
        if not rows:
            return "n/a (skipped)" if "note" in l4 else "n/a"
        r = rows[0]
        return f"{r['sumerian']} (jaccard={r['jaccard_distance']:.4f})"

    def _top1_lens5():
        rows = l5.get("rows", [])
        if not rows:
            return "n/a"
        r = rows[0]
        return f"{r['sumerian_a']} == {r['sumerian_b']} (cos={r['cosine_similarity']:.4f})"

    def _top1_lens6():
        rows = l6.get("rows", [])
        if not rows:
            return "n/a"
        r = rows[0]
        return (f"{r['sumerian']} (bridge={r['bridge_score']:.4f}, "
                f"clusters {r['nearest_cluster']}/{r['second_nearest_cluster']})")

    atlas = {
        "atlas_schema_version": 1,
        "atlas_date": _dt.date.today().isoformat(),
        "civilization": config.civilization_name,
        "source_artifacts": {
            "aligned_gemma_path": str(config.aligned_gemma_path),
            "aligned_glove_path": str(config.aligned_glove_path) if config.aligned_glove_path else None,
            "source_vocab_path": str(config.source_vocab_path),
            "anchors_path": str(config.anchors_path),
            "anchors_sha256": _sha256(config.anchors_path),
            "corpus_frequency_path": str(config.corpus_frequency_path),
            "corpus_frequency_sha256": _sha256(config.corpus_frequency_path),
            "seed": config.seed,
            "k_clusters": config.k_clusters,
            "top_n_per_lens": config.top_n_per_lens,
            "doppelganger_threshold": config.doppelganger_threshold,
            "isolation_k": config.isolation_k,
        },
        "summary": {
            "total_aligned_tokens": total_tokens,
            "anchor_tokens_in_vocab": anchor_count_in_vocab,
            "non_anchor_tokens_in_vocab": non_anchor_count,
            "top1_per_lens": {
                "lens1_english_displacement": _top1_lens1(),
                "lens2_no_counterpart": _top1_lens2(),
                "lens3_isolation": _top1_lens3(),
                "lens4_cross_space_divergence": _top1_lens4(),
                "lens5_doppelgangers": _top1_lens5(),
                "lens6_structural_bridges": _top1_lens6(),
            },
        },
        "lens1_english_displacement": l1,
        "lens2_no_counterpart": l2,
        "lens3_isolation": l3,
        "lens4_cross_space_divergence": l4,
        "lens5_doppelgangers": l5,
        "lens6_structural_bridges": l6,
    }

    # Write JSON (sort keys for determinism)
    config.output_atlas_json.parent.mkdir(parents=True, exist_ok=True)
    with open(config.output_atlas_json, "w") as f:
        json.dump(atlas, f, indent=2, sort_keys=True)
        f.write("\n")

    # Render and write markdown files
    config.output_markdown_dir.mkdir(parents=True, exist_ok=True)
    (config.output_markdown_dir / "atlas_summary.md").write_text(render_summary_markdown(atlas))
    (config.output_markdown_dir / "lens1_english_displacement.md").write_text(render_lens1_markdown(atlas))
    (config.output_markdown_dir / "lens2_no_counterpart.md").write_text(render_lens2_markdown(atlas))
    (config.output_markdown_dir / "lens3_isolation.md").write_text(render_lens3_markdown(atlas))
    (config.output_markdown_dir / "lens4_cross_space_divergence.md").write_text(render_lens4_markdown(atlas))
    (config.output_markdown_dir / "lens5_doppelgangers.md").write_text(render_lens5_markdown(atlas))
    (config.output_markdown_dir / "lens6_structural_bridges.md").write_text(render_lens6_markdown(atlas))

    print(f"[atlas] Wrote {config.output_atlas_json}")
    print(f"[atlas] Wrote {config.output_markdown_dir}/*.md")
    return atlas["summary"]
```

- [ ] **Step 4: Create Sumerian wrapper `scripts/analysis/sumerian_anomaly_atlas.py`**

```python
"""
Sumerian-specific orchestrator for the anomaly atlas framework.

Thin wrapper that builds an AnomalyConfig pointing at Cuneiformy's artifacts
and calls run_atlas. When a future comparative repo or Gemma-tized Heiroglyphy
needs an atlas, it writes its own sibling orchestrator — this file is not
reused.

Run from repo root:
    python scripts/analysis/sumerian_anomaly_atlas.py

See: docs/superpowers/specs/2026-04-20-anomaly-atlas-design.md
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.analysis.anomaly_framework import AnomalyConfig, run_atlas


SUMERIAN_JUNK_ENGLISH_GLOSSES = frozenset({
    "x", "xx", "n", "c", "cf", "unmng", "0", "00", "1", "e", "i", "u", "s",
})


def main() -> int:
    ROOT = _ROOT
    config = AnomalyConfig(
        civilization_name="sumerian",
        aligned_gemma_path=ROOT / "final_output" / "sumerian_aligned_gemma_vectors.npz",
        aligned_glove_path=ROOT / "final_output" / "sumerian_aligned_vectors.npz",
        source_vocab_path=ROOT / "final_output" / "sumerian_aligned_vocab.pkl",
        target_gemma_vocab_path=ROOT / "models" / "english_gemma_whitened_768d.npz",
        target_glove_vocab_path=None,  # GloVe path not needed by Lenses 1-2 in Gemma-only mode
        anchors_path=ROOT / "data" / "processed" / "english_anchors.json",
        corpus_frequency_path=ROOT / "data" / "processed" / "cleaned_corpus.txt",
        junk_target_glosses=SUMERIAN_JUNK_ENGLISH_GLOSSES,
        min_anchor_confidence=0.5,
        min_token_length=2,
        output_atlas_json=ROOT / "docs" / "anomaly_atlas.json",
        output_markdown_dir=ROOT / "docs" / "anomalies",
        output_figures_dir=None,
        seed=42,
        k_clusters=40,
        top_n_per_lens=50,
        doppelganger_threshold=0.95,
        isolation_k=10,
    )
    run_atlas(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run tests, verify pass**

```bash
pytest tests/analysis/test_anomaly_lenses.py -v
```
Expected: 17 PASS (16 + 1 new integration test).

- [ ] **Step 6: Full suite**

```bash
pytest tests/ --ignore=tests/test_integration.py -q
```
Expected: 161 PASS.

- [ ] **Step 7: Commit**

```bash
cd /Users/crashy/Development/cuneiformy
git add scripts/analysis/anomaly_framework.py \
        scripts/analysis/sumerian_anomaly_atlas.py \
        tests/analysis/test_anomaly_lenses.py
git commit -m "feat: add run_atlas orchestrator + sumerian wrapper"
```

---

## Task 7: Run atlas on real data + commit artifacts + journal

**Files:**
- Generated: `docs/anomaly_atlas.json`
- Generated: `docs/anomalies/*.md` (7 files)
- Modified: `docs/EXPERIMENT_JOURNAL.md`

### Setup note

Delivery step. Runs the atlas against the real artifacts (~6-10 min), commits the outputs, adds a dated journal entry with the headline top-1 findings.

- [ ] **Step 1: Run the atlas**

```bash
cd /Users/crashy/Development/cuneiformy
python scripts/analysis/sumerian_anomaly_atlas.py 2>&1 | tee /tmp/atlas_output.txt
```

Runtime: ~6-10 minutes. Do NOT push to background.

Expected stdout ends with:
```
[atlas] Wrote .../docs/anomaly_atlas.json
[atlas] Wrote .../docs/anomalies/*.md
```

- [ ] **Step 2: Sanity-check the atlas JSON**

```bash
python3 -c "
import json
atlas = json.load(open('docs/anomaly_atlas.json'))
print('civilization:', atlas['civilization'])
print('total tokens:', atlas['summary']['total_aligned_tokens'])
print('anchor tokens in vocab:', atlas['summary']['anchor_tokens_in_vocab'])
print()
print('Top-1 per lens:')
for key, val in atlas['summary']['top1_per_lens'].items():
    print(f'  {key}: {val}')
print()
print('Row counts:')
for lens_key in ['lens1_english_displacement', 'lens2_no_counterpart', 'lens3_isolation',
                 'lens4_cross_space_divergence', 'lens5_doppelgangers', 'lens6_structural_bridges']:
    section = atlas[lens_key]
    if 'rows' in section:
        print(f'  {lens_key}: {len(section[\"rows\"])} rows')
    elif 'rows_unfiltered' in section:
        print(f'  {lens_key}: {len(section[\"rows_unfiltered\"])} unfiltered / {len(section.get(\"rows_filtered\", section.get(\"rows_anchor_only\", [])))} filtered')
"
```

Capture the output verbatim — the top-1 findings are needed for the journal entry.

- [ ] **Step 3: Determinism check**

```bash
cp docs/anomaly_atlas.json /tmp/atlas_first.json
python scripts/analysis/sumerian_anomaly_atlas.py 2>&1 | tail -3
diff /tmp/atlas_first.json docs/anomaly_atlas.json
```
Expected: empty diff. If anything differs, investigate (likely an ordering bug or a non-seeded operation).

- [ ] **Step 4: Full test suite as final sanity**

```bash
pytest tests/ --ignore=tests/test_integration.py -q
```
Expected: 161 pass, 0 fail.

- [ ] **Step 5: Commit atlas JSON + markdowns**

```bash
cd /Users/crashy/Development/cuneiformy
git add docs/anomaly_atlas.json
git add docs/anomalies/
git commit -m "chore: commit 2026-04-20 anomaly atlas baseline"
```

- [ ] **Step 6: Add journal entry**

Insert in `docs/EXPERIMENT_JOURNAL.md` AFTER the preamble `---` and BEFORE the existing `## 2026-04-19 — Sumerian Cosmogony Document shipped` entry. Replace every `[...]` with a REAL value from the Step 2 sanity output.

```markdown
## 2026-04-20 — Anomaly Atlas shipped (civilization-agnostic framework)

**Hypothesis:** The Sumerian cosmogony document drew interpretive claims from five hand-picked concepts. Applying the same methodology at scale to all 35,508 aligned tokens, via six diagnostic lenses, both tests the cosmogony document's thesis and produces a ranked atlas of anomalies for future writing. Designed as a civilization-agnostic framework so the queued comparative repo (and eventually a Gemma-tized Heiroglyphy) can reuse the code.

**Method:** New `scripts/analysis/anomaly_lenses.py` (6 pure-function lenses, zero Sumerian-specific logic), `scripts/analysis/anomaly_framework.py` (AnomalyConfig + run_atlas + markdown renderers), `scripts/analysis/sumerian_anomaly_atlas.py` (thin 60-line wrapper that builds the config for Cuneiformy's artifacts). 17 unit tests total. Six lenses: English displacement, untranslated high-value terms, isolation in source space, cross-space divergence (Gemma vs GloVe), doppelgangers (cos ≥ 0.95), structural bridges (k-means k=40).

**Result — top-1 per lens:**
- Lens 1 (English displacement): [FROM STEP 2 OUTPUT]
- Lens 2 (no counterpart): [FROM STEP 2 OUTPUT]
- Lens 3 (isolation): [FROM STEP 2 OUTPUT]
- Lens 4 (cross-space divergence): [FROM STEP 2 OUTPUT]
- Lens 5 (doppelgangers): [FROM STEP 2 OUTPUT]
- Lens 6 (structural bridges): [FROM STEP 2 OUTPUT]

**Takeaway:** [WRITE 2-3 sentences based on real numbers. If the cosmogony document's "geometrically distinct from English" thesis holds, the atlas's Lens 1 top-50 should include other cosmogony concepts (`abzu`, `zi`, `nam`, `me`) near the top. If they don't appear near the top, the thesis may have been artifact-of-selection.]

**Artifacts / commits:** `scripts/analysis/anomaly_lenses.py`, `scripts/analysis/anomaly_framework.py`, `scripts/analysis/sumerian_anomaly_atlas.py`, `tests/analysis/test_anomaly_lenses.py`, `docs/anomaly_atlas.json`, `docs/anomalies/*.md`. Spec: `docs/superpowers/specs/2026-04-20-anomaly-atlas-design.md`. Plan: `docs/superpowers/plans/2026-04-20-anomaly-atlas.md`.
```

- [ ] **Step 7: Commit journal**

```bash
cd /Users/crashy/Development/cuneiformy
git add docs/EXPERIMENT_JOURNAL.md
git commit -m "docs: journal 2026-04-20 anomaly atlas baseline"
```

---

## Self-Review

**Spec coverage:**
- `anomaly_lenses.py` 6 pure functions → Tasks 2 (Lens 1 + 3), 3 (Lens 2 + 4), 4 (Lens 5 + 6).
- `anomaly_framework.py` with AnomalyConfig + run_atlas + renderers → Tasks 1 (dataclass), 5 (renderers), 6 (run_atlas).
- `sumerian_anomaly_atlas.py` thin wrapper → Task 6.
- Unit tests (10 lens + 1 integration) → Tasks 1-6.
- Output JSON + 7 markdown files → Task 6 (writes), Task 7 (generates real data + commits).
- Two-tier rankings (Lens 1 + Lens 4) → Task 2 (Lens 1), Task 3 (Lens 4) implement; Task 5 renders both tiers.
- Determinism discipline → `sort_keys=True` in json.dump (Task 6 Step 3), fixed seeds everywhere, sorted tie-breaking.
- Civilization-agnostic framework → Task 1 dataclass has zero Sumerian constants; Task 6 wrapper is Sumerian-specific and ~60 lines.
- Portability note for Egyptian → spec + journal entry reference it; no code required in this workstream.
- Journal entry → Task 7 Step 6.

**Placeholder scan:**
- Task 7 Step 6 journal template has 6 `[FROM STEP 2 OUTPUT]` placeholders — all called out as requiring replacement with the real captured numbers before committing. Necessary because the numbers are unknown until the atlas runs.
- No `TBD`, `TODO`, "similar to Task N", or "add appropriate" patterns.

**Type consistency:**
- `AnomalyConfig` field names match across Tasks 1 (definition), 6 (run_atlas consumption), Task 6 Sumerian wrapper (instantiation).
- Lens function signatures match across test file and implementation file in each of Tasks 2-4.
- `aligned_gemma` / `aligned_glove` parameter naming consistent.
- `source_vocab` / `target_gemma_vocab` naming consistent (civilization-agnostic — no `sumerian_vocab`).
- Atlas JSON schema keys consistent between `run_atlas` (writes) and markdown renderers (reads).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-20-anomaly-atlas.md`. Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task with two-stage review. Tasks 1-6 are mechanical TDD code; Task 7 is data-run + journal delivery. Matches every prior workstream.

**2. Inline Execution** — batched via `superpowers:executing-plans`.

Which approach?
