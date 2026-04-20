# Sumerian Cosmogony Document Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce `docs/sumerian_cosmogony.md` — a ~10,500-word methodology-rigorous case study on Sumerian cosmogony using the 52%-top-1 whitened-Gemma alignment. Plus the supporting `scripts/analysis/` infrastructure (6 modules, 2 entry points), 7 committed PNG figures, 1 canonical JSON tables file, pre-flight concept-availability check, and full test coverage.

**Architecture:** Six small focused analysis modules with clear responsibilities (pairwise distances, English displacement, ETCSL passage retrieval, UMAP projection, pre-flight check), composed by two deterministic top-level entry points (figures + tables). Document prose is hand-written after the generated artifacts exist, so numeric claims trace to data. Pre-flight runs first and gates concept selection before prose is written.

**Tech Stack:** Python 3, numpy, matplotlib, umap-learn (new dep), gensim (existing), pytest. No new models, no new corpora, no pipeline changes.

**Reference spec:** `docs/superpowers/specs/2026-04-19-sumerian-cosmogony-document-design.md`

---

## Before You Begin

- Current branch: `master`. Cut a fresh feature branch:
  ```bash
  cd /Users/crashy/Development/cuneiformy
  git checkout -b feat/cosmogony-document
  ```
  All commits land on `feat/cosmogony-document`. Merge via `superpowers:finishing-a-development-branch` after Task 7.

- Verify input artifacts are locally present:
  ```bash
  ls -la \
    final_output/sumerian_aligned_gemma_vectors.npz \
    final_output/sumerian_aligned_vectors.npz \
    final_output/sumerian_aligned_vocab.pkl \
    models/english_gemma_whitened_768d.npz \
    data/processed/glove.6B.300d.txt \
    data/raw/etcsl_texts.json
  ```
  These are the artifacts Workstream 2b produced; they must exist before Task 5.

- New dependency: `umap-learn`. Add to `requirements.txt` and `pip install` in Task 3.

---

## File Structure

**New files:**
- `scripts/analysis/__init__.py` (empty)
- `scripts/analysis/semantic_field.py`
- `scripts/analysis/english_displacement.py`
- `scripts/analysis/etcsl_passage_finder.py`
- `scripts/analysis/umap_projection.py`
- `scripts/analysis/preflight_concept_check.py`
- `scripts/analysis/generate_cosmogony_tables.py` (entry)
- `scripts/analysis/generate_cosmogony_figures.py` (entry)
- `scripts/analysis/cosmogony_concepts.py` (shared concept-slate config)
- `tests/analysis/__init__.py` (empty)
- `tests/analysis/test_semantic_field.py`
- `tests/analysis/test_english_displacement.py`
- `tests/analysis/test_etcsl_passage_finder.py`
- `tests/analysis/test_umap_projection.py`
- `tests/analysis/test_preflight_concept_check.py`
- `docs/sumerian_cosmogony.md`
- `docs/cosmogony_tables.json`
- `docs/figures/cosmogony/*.png` (7 files, generated)
- `results/cosmogony_preflight_2026-04-19.json`

**Modified files:**
- `requirements.txt` — add umap-learn pin.
- `docs/EXPERIMENT_JOURNAL.md` — journal entry upon completion.
- `README.md` — add a link to the document in the Research Progress section.

**Untouched:**
- All pipeline scripts (01-10, validate_phase_b, audit_anchors, coverage_diagnostic, sumerian_normalize).
- All existing tests.
- `final_output/sumerian_lookup.py` (the dual-view API already has every method we need).
- All model files.

---

## Task 1: Scaffolding + `semantic_field.py` (TDD)

**Files:**
- Create: `scripts/analysis/__init__.py`
- Create: `scripts/analysis/semantic_field.py`
- Create: `tests/analysis/__init__.py`
- Create: `tests/analysis/test_semantic_field.py`

### Setup note

`semantic_field.py` provides two functions: `compute_pairwise_distances` (N×N cosine-distance matrix over Sumerian tokens, via the aligned vectors loaded through `SumerianLookup`'s `_spaces[space]["sum_norm"]` internal state) and `render_semantic_field_heatmap` (matplotlib heatmap → PNG).

Cosine distance = 1 - cosine similarity. We use distance (not similarity) so the heatmap reads naturally: dark = closer, light = farther.

- [ ] **Step 1: Write failing tests**

Create `tests/analysis/__init__.py` (empty). Create `scripts/analysis/__init__.py` (empty).

Create `tests/analysis/test_semantic_field.py`:

```python
import tempfile
from pathlib import Path

import numpy as np
import pytest


def _synthetic_lookup(tokens, vectors):
    """Minimal SumerianLookup stub exposing the _spaces dict our functions use."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = vectors / norms

    class Stub:
        def __init__(self):
            self.vocab = tokens
            self._spaces = {
                "gemma": {"sum_norm": normalized, "sum_dim": vectors.shape[1]},
            }
            self._token_to_idx = {t: i for i, t in enumerate(tokens)}
    return Stub()


def test_pairwise_distances_is_symmetric():
    from scripts.analysis.semantic_field import compute_pairwise_distances

    tokens = ["a", "b", "c"]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    lookup = _synthetic_lookup(tokens, vectors)

    D = compute_pairwise_distances(lookup, tokens, space="gemma")
    assert np.allclose(D, D.T)


def test_pairwise_distances_diagonal_is_zero():
    from scripts.analysis.semantic_field import compute_pairwise_distances

    tokens = ["a", "b"]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    lookup = _synthetic_lookup(tokens, vectors)

    D = compute_pairwise_distances(lookup, tokens, space="gemma")
    assert np.allclose(np.diag(D), 0.0)


def test_pairwise_distances_shape_matches_input():
    from scripts.analysis.semantic_field import compute_pairwise_distances

    tokens = ["a", "b", "c", "d"]
    vectors = np.random.default_rng(0).standard_normal((4, 8)).astype(np.float32)
    lookup = _synthetic_lookup(tokens, vectors)

    D = compute_pairwise_distances(lookup, tokens, space="gemma")
    assert D.shape == (4, 4)


def test_pairwise_distances_raises_on_unknown_token():
    from scripts.analysis.semantic_field import compute_pairwise_distances

    tokens = ["a", "b"]
    vectors = np.eye(2, dtype=np.float32)
    lookup = _synthetic_lookup(tokens, vectors)

    with pytest.raises(KeyError, match="unknown"):
        compute_pairwise_distances(lookup, ["a", "unknown"], space="gemma")


def test_heatmap_writes_png():
    from scripts.analysis.semantic_field import render_semantic_field_heatmap

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "heatmap.png"
        distances = np.array([[0.0, 0.3, 0.5], [0.3, 0.0, 0.2], [0.5, 0.2, 0.0]])
        render_semantic_field_heatmap(
            distances, ["a", "b", "c"], title="test", out_path=out_path
        )
        assert out_path.exists()
        assert out_path.stat().st_size > 0
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
cd /Users/crashy/Development/cuneiformy
pytest tests/analysis/test_semantic_field.py -v
```
Expected: 5 tests FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `scripts/analysis/semantic_field.py`**

```python
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
```

- [ ] **Step 4: Run tests, verify all pass**

```bash
pytest tests/analysis/test_semantic_field.py -v
```
Expected: 5 pass.

- [ ] **Step 5: Run full test suite for regression check**

```bash
pytest tests/ --ignore=tests/test_integration.py -q
```
Expected: 125 pass (120 prior + 5 new).

- [ ] **Step 6: Commit**

```bash
cd /Users/crashy/Development/cuneiformy
git add scripts/analysis/__init__.py scripts/analysis/semantic_field.py \
        tests/analysis/__init__.py tests/analysis/test_semantic_field.py
git commit -m "feat: add semantic_field analysis module with heatmap renderer"
```

---

## Task 2: `english_displacement.py` + `etcsl_passage_finder.py` (TDD)

**Files:**
- Create: `scripts/analysis/english_displacement.py`
- Create: `scripts/analysis/etcsl_passage_finder.py`
- Create: `tests/analysis/test_english_displacement.py`
- Create: `tests/analysis/test_etcsl_passage_finder.py`

### Setup note

These two modules are independent. `english_displacement` measures how far a Sumerian-projected vector lands from the English seed's native vector in the same target space. `etcsl_passage_finder` retrieves ETCSL passages containing a given normalized Sumerian token for §7 of each deep dive.

Both use tiny synthetic inputs — no dependence on real alignment artifacts for the unit tests.

- [ ] **Step 1: Write the failing tests**

Create `tests/analysis/test_english_displacement.py`:

```python
import numpy as np
import pytest


def _displacement_lookup_stub(eng_vocab, eng_vectors, sum_vocab, sum_vectors_gemma):
    """SumerianLookup stub exposing _spaces with eng_norm and sum_norm."""
    def _norm(X):
        n = np.linalg.norm(X, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return X / n

    class Stub:
        def __init__(self):
            self.vocab = sum_vocab
            self._spaces = {
                "gemma": {
                    "sum_norm": _norm(sum_vectors_gemma),
                    "sum_dim": sum_vectors_gemma.shape[1],
                    "eng_norm": _norm(eng_vectors),
                    "eng_vocab_map": {w.lower(): i for i, w in enumerate(eng_vocab)},
                }
            }
            self._token_to_idx = {t: i for i, t in enumerate(sum_vocab)}
    return Stub()


def test_displacement_computes_cosine():
    from scripts.analysis.english_displacement import english_displacement

    eng_vocab = ["king"]
    eng_vectors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    sum_vocab = ["lugal"]
    sum_vectors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)  # identical direction

    lookup = _displacement_lookup_stub(eng_vocab, eng_vectors, sum_vocab, sum_vectors)

    result = english_displacement(lookup, "lugal", "king", space="gemma")
    assert result["cosine_similarity"] == pytest.approx(1.0, abs=1e-5)
    assert result["cosine_distance"] == pytest.approx(0.0, abs=1e-5)


def test_displacement_orthogonal_vectors():
    from scripts.analysis.english_displacement import english_displacement

    eng_vocab = ["god"]
    eng_vectors = np.array([[1.0, 0.0]], dtype=np.float32)
    sum_vocab = ["dingir"]
    sum_vectors = np.array([[0.0, 1.0]], dtype=np.float32)
    lookup = _displacement_lookup_stub(eng_vocab, eng_vectors, sum_vocab, sum_vectors)

    result = english_displacement(lookup, "dingir", "god", space="gemma")
    assert result["cosine_similarity"] == pytest.approx(0.0, abs=1e-5)


def test_displacement_raises_on_unknown_sumerian():
    from scripts.analysis.english_displacement import english_displacement

    lookup = _displacement_lookup_stub(
        ["king"], np.eye(1, 2, dtype=np.float32),
        ["lugal"], np.eye(1, 2, dtype=np.float32),
    )
    with pytest.raises(KeyError):
        english_displacement(lookup, "missing", "king", space="gemma")


def test_displacement_raises_on_unknown_english():
    from scripts.analysis.english_displacement import english_displacement

    lookup = _displacement_lookup_stub(
        ["king"], np.eye(1, 2, dtype=np.float32),
        ["lugal"], np.eye(1, 2, dtype=np.float32),
    )
    with pytest.raises(KeyError):
        english_displacement(lookup, "lugal", "unknown", space="gemma")


def test_displacement_is_case_insensitive_on_english():
    from scripts.analysis.english_displacement import english_displacement

    lookup = _displacement_lookup_stub(
        ["king"], np.eye(1, 2, dtype=np.float32),
        ["lugal"], np.eye(1, 2, dtype=np.float32),
    )
    r1 = english_displacement(lookup, "lugal", "king", space="gemma")
    r2 = english_displacement(lookup, "lugal", "KING", space="gemma")
    assert r1["cosine_similarity"] == pytest.approx(r2["cosine_similarity"])
```

Create `tests/analysis/test_etcsl_passage_finder.py`:

```python
import pytest


def test_returns_passages_with_token():
    from scripts.analysis.etcsl_passage_finder import find_passages

    etcsl = [
        {
            "text_id": "t.1.2",
            "title": "Enki and Ninmah",
            "lines": [
                {"line_no": 1, "transliteration": "lugal an ki", "translation": "king of heaven and earth"},
                {"line_no": 2, "transliteration": "nam-tar gal", "translation": "great fate"},
                {"line_no": 3, "transliteration": "enki dugal", "translation": "wise Enki"},
            ],
        },
    ]

    passages = find_passages("nam-tar", etcsl, max_passages=3, context_lines=1)
    assert len(passages) == 1
    assert passages[0]["text_id"] == "t.1.2"
    assert passages[0]["matched_line_no"] == 2


def test_respects_max_passages_limit():
    from scripts.analysis.etcsl_passage_finder import find_passages

    etcsl = [
        {
            "text_id": f"t.{i}",
            "title": f"Text {i}",
            "lines": [
                {"line_no": 1, "transliteration": "nam-tar x", "translation": "fate"},
            ],
        }
        for i in range(10)
    ]

    passages = find_passages("nam-tar", etcsl, max_passages=3)
    assert len(passages) == 3


def test_zero_passages_when_token_absent():
    from scripts.analysis.etcsl_passage_finder import find_passages

    etcsl = [
        {"text_id": "t.1", "title": "test", "lines": [
            {"line_no": 1, "transliteration": "lugal", "translation": "king"}
        ]}
    ]
    assert find_passages("me", etcsl) == []


def test_context_lines_captured():
    from scripts.analysis.etcsl_passage_finder import find_passages

    etcsl = [
        {
            "text_id": "t.1",
            "title": "t",
            "lines": [
                {"line_no": 1, "transliteration": "line one", "translation": "l1"},
                {"line_no": 2, "transliteration": "line two", "translation": "l2"},
                {"line_no": 3, "transliteration": "nam-tar word", "translation": "fate"},
                {"line_no": 4, "transliteration": "line four", "translation": "l4"},
                {"line_no": 5, "transliteration": "line five", "translation": "l5"},
            ],
        }
    ]

    passages = find_passages("nam-tar", etcsl, max_passages=1, context_lines=2)
    assert len(passages) == 1
    # context includes 2 lines before and 2 lines after the matched line.
    assert len(passages[0]["context"]) == 5
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
pytest tests/analysis/test_english_displacement.py tests/analysis/test_etcsl_passage_finder.py -v
```
Expected: 5 + 4 = 9 FAIL on ImportError.

- [ ] **Step 3: Implement both modules**

Create `scripts/analysis/english_displacement.py`:

```python
"""
English-displacement measurement.

Measures how far a Sumerian-projected vector lands from the English seed's
native vector in the same target space. Reported per concept in §6 of each
deep dive.

See: docs/superpowers/specs/2026-04-19-sumerian-cosmogony-document-design.md
"""
from __future__ import annotations

import numpy as np


def english_displacement(
    lookup,
    sumerian_token: str,
    english_seed: str,
    space: str = "gemma",
) -> dict:
    """Return cosine between Sumerian-projected and English-native vectors.

    Both vectors are pre-L2-normalized in the SumerianLookup, so the dot
    product equals cosine similarity.
    """
    s = lookup._spaces[space]

    # Sumerian side
    idx_map = {t: i for i, t in enumerate(lookup.vocab)}
    if sumerian_token not in idx_map:
        raise KeyError(f"unknown Sumerian token: {sumerian_token!r}")
    sum_vec = s["sum_norm"][idx_map[sumerian_token]]

    # English side
    eng_lower = english_seed.lower()
    if eng_lower not in s["eng_vocab_map"]:
        raise KeyError(f"unknown English seed in {space!r} vocab: {english_seed!r}")
    eng_vec = s["eng_norm"][s["eng_vocab_map"][eng_lower]]

    cos_sim = float(np.clip(np.dot(sum_vec, eng_vec), -1.0, 1.0))
    return {
        "sumerian_token": sumerian_token,
        "english_seed": english_seed,
        "space": space,
        "cosine_similarity": cos_sim,
        "cosine_distance": 1.0 - cos_sim,
    }
```

Create `scripts/analysis/etcsl_passage_finder.py`:

```python
"""
ETCSL passage retrieval for per-concept source-text grounding.

Given a Sumerian token (normalized), find 1-N passages in
data/raw/etcsl_texts.json where the token appears, returning the matched
line plus surrounding context lines.

Used in §7 of each deep dive to connect geometric claims to textual reality.

See: docs/superpowers/specs/2026-04-19-sumerian-cosmogony-document-design.md
"""
from __future__ import annotations


def find_passages(
    sumerian_token: str,
    etcsl_texts: list[dict],
    max_passages: int = 3,
    context_lines: int = 2,
) -> list[dict]:
    """Find ETCSL passages containing the token.

    Returns a list of {text_id, title, matched_line_no, transliteration,
    translation, context} dicts. `context` is [context_lines before] +
    matched line + [context_lines after].
    """
    results = []
    for text in etcsl_texts:
        lines = text.get("lines", [])
        for i, line in enumerate(lines):
            trans = line.get("transliteration", "") or ""
            # Whitespace-split to match whole tokens (avoids false positives
            # inside compound words the way substring-search would).
            if sumerian_token in trans.split():
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                results.append({
                    "text_id": text.get("text_id"),
                    "title": text.get("title", ""),
                    "matched_line_no": line.get("line_no", i),
                    "transliteration": trans,
                    "translation": line.get("translation", ""),
                    "context": lines[start:end],
                })
                if len(results) >= max_passages:
                    return results
                break  # one match per text
    return results
```

- [ ] **Step 4: Run tests, verify all pass**

```bash
pytest tests/analysis/test_english_displacement.py tests/analysis/test_etcsl_passage_finder.py -v
```
Expected: 9 pass.

- [ ] **Step 5: Full suite regression check**

```bash
pytest tests/ --ignore=tests/test_integration.py -q
```
Expected: 134 pass (125 + 9 new).

- [ ] **Step 6: Commit**

```bash
cd /Users/crashy/Development/cuneiformy
git add scripts/analysis/english_displacement.py \
        scripts/analysis/etcsl_passage_finder.py \
        tests/analysis/test_english_displacement.py \
        tests/analysis/test_etcsl_passage_finder.py
git commit -m "feat: add english_displacement + etcsl_passage_finder analysis modules"
```

---

## Task 3: `umap_projection.py` + dependency (TDD)

**Files:**
- Modify: `requirements.txt` (add `umap-learn`).
- Create: `scripts/analysis/umap_projection.py`
- Create: `tests/analysis/test_umap_projection.py`

### Setup note

UMAP needs `n_neighbors < n_samples`. For tiny token lists (<15), UMAP can be unstable; fall back to PCA 2D in that case and note the fallback in the figure caption.

- [ ] **Step 1: Add umap-learn to requirements.txt**

Read current requirements.txt, append `umap-learn>=0.5.5` to the dependency list.

Then install:

```bash
cd /Users/crashy/Development/cuneiformy
pip install "umap-learn>=0.5.5"
```

- [ ] **Step 2: Write failing tests**

Create `tests/analysis/test_umap_projection.py`:

```python
import tempfile
from pathlib import Path

import numpy as np


def _umap_lookup_stub(tokens, vectors):
    def _norm(X):
        n = np.linalg.norm(X, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return X / n

    class Stub:
        def __init__(self):
            self.vocab = tokens
            self._spaces = {
                "gemma": {"sum_norm": _norm(vectors), "sum_dim": vectors.shape[1]}
            }
    return Stub()


def test_umap_writes_png_for_sufficient_vocab():
    from scripts.analysis.umap_projection import umap_cosmogonic_vocabulary

    rng = np.random.default_rng(0)
    tokens = [f"t{i}" for i in range(20)]
    vectors = rng.standard_normal((20, 16)).astype(np.float32)
    lookup = _umap_lookup_stub(tokens, vectors)
    labels = {t: "cluster_a" if i % 2 == 0 else "cluster_b" for i, t in enumerate(tokens)}

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "umap.png"
        umap_cosmogonic_vocabulary(lookup, tokens, labels, space="gemma", out_path=out_path)
        assert out_path.exists()
        assert out_path.stat().st_size > 0


def test_umap_falls_back_to_pca_when_vocab_too_small():
    from scripts.analysis.umap_projection import umap_cosmogonic_vocabulary

    tokens = ["a", "b", "c"]  # too small for UMAP
    vectors = np.eye(3, 8, dtype=np.float32)
    lookup = _umap_lookup_stub(tokens, vectors)
    labels = {"a": "x", "b": "x", "c": "y"}

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "fallback.png"
        umap_cosmogonic_vocabulary(lookup, tokens, labels, space="gemma", out_path=out_path)
        assert out_path.exists()


def test_umap_deterministic_with_fixed_seed():
    """Two calls with the same inputs should produce the same PNG bytes."""
    from scripts.analysis.umap_projection import umap_cosmogonic_vocabulary

    rng = np.random.default_rng(42)
    tokens = [f"t{i}" for i in range(20)]
    vectors = rng.standard_normal((20, 16)).astype(np.float32)
    lookup = _umap_lookup_stub(tokens, vectors)
    labels = {t: "a" for t in tokens}

    with tempfile.TemporaryDirectory() as tmp:
        p1 = Path(tmp) / "one.png"
        p2 = Path(tmp) / "two.png"
        umap_cosmogonic_vocabulary(lookup, tokens, labels, space="gemma", out_path=p1)
        umap_cosmogonic_vocabulary(lookup, tokens, labels, space="gemma", out_path=p2)
        # UMAP with fixed random_state + same matplotlib backend should be deterministic.
        # PNG bytes may vary due to matplotlib version; we test that the embedding
        # coords are identical by running UMAP twice explicitly.
        # (The PNG test above proves the file is produced; this test proves the
        # coordinate computation is stable.)
        from scripts.analysis.umap_projection import _compute_embedding
        coords1 = _compute_embedding(
            np.linalg.norm(vectors, axis=1, keepdims=True).clip(1e-9) * 0 + vectors / np.linalg.norm(vectors, axis=1, keepdims=True).clip(1e-9),
            seed=42,
        )
        coords2 = _compute_embedding(
            vectors / np.linalg.norm(vectors, axis=1, keepdims=True).clip(1e-9),
            seed=42,
        )
        assert np.allclose(coords1, coords2)
```

- [ ] **Step 3: Run tests, verify they fail**

```bash
pytest tests/analysis/test_umap_projection.py -v
```
Expected: 3 FAIL on ImportError.

- [ ] **Step 4: Implement `scripts/analysis/umap_projection.py`**

```python
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
```

- [ ] **Step 5: Run tests, verify all pass**

```bash
pytest tests/analysis/test_umap_projection.py -v
```
Expected: 3 pass.

- [ ] **Step 6: Full suite regression**

```bash
pytest tests/ --ignore=tests/test_integration.py -q
```
Expected: 137 pass.

- [ ] **Step 7: Commit**

```bash
cd /Users/crashy/Development/cuneiformy
git add requirements.txt \
        scripts/analysis/umap_projection.py \
        tests/analysis/test_umap_projection.py
git commit -m "feat: add umap_projection analysis module with PCA fallback"
```

---

## Task 4: `preflight_concept_check.py` + run pre-flight + lock concept slate (TDD + data gate)

**Files:**
- Create: `scripts/analysis/cosmogony_concepts.py` — the canonical concept slate.
- Create: `scripts/analysis/preflight_concept_check.py`
- Create: `tests/analysis/test_preflight_concept_check.py`
- Create: `results/cosmogony_preflight_2026-04-19.json` (via running the pre-flight)

### Setup note

Pre-flight is a data gate: after this task, the concept slate is locked or substituted. The `cosmogony_concepts.py` config file holds the slate so later scripts (generators) import it rather than hardcoding.

- [ ] **Step 1: Create the concept-slate config**

Create `scripts/analysis/cosmogony_concepts.py`:

```python
"""
Canonical Sumerian cosmogony concept slate for the document.

Each concept pairs a Sumerian token with its English seed and thematic tag.
Pre-flight check in preflight_concept_check.py validates each against the
current SumerianLookup + ETCSL corpus, flagging substitutions if needed.

See: docs/superpowers/specs/2026-04-19-sumerian-cosmogony-document-design.md
"""

PRIMARY_CONCEPTS = [
    {"sumerian": "abzu",    "english": "deep",  "theme": "primordial"},
    {"sumerian": "zi",      "english": "breath", "theme": "animation"},
    {"sumerian": "nam",     "english": "essence", "theme": "naming"},
    {"sumerian": "namtar",  "english": "fate",  "theme": "decree"},
    {"sumerian": "me",      "english": "decree", "theme": "civilization"},
]

# Substitutes drawn on if a primary concept fails pre-flight.
ALTERNATE_CONCEPTS = [
    {"sumerian": "ima",     "english": "clay",     "theme": "matter"},
    {"sumerian": "kur",     "english": "mountain", "theme": "netherworld"},
    {"sumerian": "an",      "english": "heaven",   "theme": "primordial"},
    {"sumerian": "ki",      "english": "earth",    "theme": "primordial"},
]

# Anunnaki-adjacent vocabulary for the narrative-spine UMAP figure (§3).
ANUNNAKI_VOCABULARY = [
    "an", "ki", "enki", "enlil", "nammu", "ninmah", "inanna", "utu", "nanna",
    "nam", "me", "namtar", "zi", "abzu", "ima", "kur", "dingir", "lugal",
]

# Tokens used to define the "cosmogonic axis" in the §9 synthesis figure.
COSMOGONIC_POLES = {
    "primordial_pole": ["abzu", "nammu"],     # pre-creation
    "decree_pole": ["me", "namtar", "dingir"],  # civilizational order
}
```

Note: Sumerian tokens are written WITHOUT hyphens (e.g., `namtar` not `nam-tar`) because the Workstream 2b normalization chain drops hyphens. Prose in the document will use the conventional `nam-tar` form, while code uses the normalized `namtar`. This is documented in the doc's methodology section.

- [ ] **Step 2: Write failing tests**

Create `tests/analysis/test_preflight_concept_check.py`:

```python
import pytest


def _preflight_stub():
    """SumerianLookup stub with a known vocab and find_both behavior."""
    import numpy as np

    class Stub:
        def __init__(self):
            self.vocab = ["abzu", "zi", "nam", "namtar", "me", "ima", "kur"]
            self._eng_vocabs = {
                "gemma": {"deep", "breath", "essence", "fate", "decree", "clay", "mountain"},
                "glove": {"deep", "breath", "essence", "fate", "decree", "clay", "mountain"},
            }

        def find(self, english_word, top_k=10, space="gemma"):
            # Return a plausible top-K for known English words, else [].
            eng = english_word.lower()
            if eng not in self._eng_vocabs.get(space, set()):
                return []
            # Non-degenerate by default: return 5 multi-char Sumerian words.
            return [(w, 0.5 - i * 0.05) for i, w in enumerate(self.vocab[:5])]

        def find_both(self, english_word, top_k=10):
            return {
                "gemma": self.find(english_word, top_k=top_k, space="gemma"),
                "glove": self.find(english_word, top_k=top_k, space="glove"),
            }
    return Stub()


def test_passes_when_concept_fully_resolved():
    from scripts.analysis.preflight_concept_check import preflight_check

    lookup = _preflight_stub()
    concepts = [{"sumerian": "abzu", "english": "deep", "theme": "primordial"}]
    etcsl = [
        {"text_id": "t.1", "title": "", "lines": [
            {"line_no": 1, "transliteration": "abzu gal", "translation": "great deep"}
        ]}
    ]
    report = preflight_check(lookup, concepts, etcsl)
    verdict = report["concepts"][0]
    assert verdict["status"] == "pass"
    assert verdict["sumerian_in_vocab"] is True
    assert verdict["english_in_gemma"] is True
    assert verdict["etcsl_passages"] >= 1


def test_flags_vocab_miss():
    from scripts.analysis.preflight_concept_check import preflight_check

    lookup = _preflight_stub()
    concepts = [{"sumerian": "nonexistent", "english": "deep", "theme": "x"}]
    report = preflight_check(lookup, concepts, [])
    verdict = report["concepts"][0]
    assert verdict["status"] == "fail"
    assert verdict["sumerian_in_vocab"] is False
    assert "sumerian_vocab_miss" in verdict["failure_reasons"]


def test_flags_degenerate_top5():
    from scripts.analysis.preflight_concept_check import preflight_check

    # Build a lookup whose find() returns single-char degenerate results.
    class DegenerateStub:
        vocab = ["abzu", "a", "b", "c", "d", "e"]
        def find(self, english_word, top_k=10, space="gemma"):
            return [("a", 0.9), ("b", 0.8), ("c", 0.7), ("abzu", 0.6), ("d", 0.5)]
        def find_both(self, english_word, top_k=10):
            return {"gemma": self.find(english_word, top_k, "gemma"),
                    "glove": self.find(english_word, top_k, "glove")}
    lookup = DegenerateStub()

    concepts = [{"sumerian": "abzu", "english": "deep", "theme": "primordial"}]
    # Provide matching etcsl so only the degenerate-top5 flag trips.
    etcsl = [{"text_id": "t.1", "title": "",
              "lines": [{"line_no": 1, "transliteration": "abzu", "translation": "deep"}]}]
    report = preflight_check(lookup, concepts, etcsl)
    verdict = report["concepts"][0]
    assert "degenerate_top5" in verdict["warnings"]


def test_flags_zero_etcsl_passages():
    from scripts.analysis.preflight_concept_check import preflight_check

    lookup = _preflight_stub()
    concepts = [{"sumerian": "abzu", "english": "deep", "theme": "primordial"}]
    report = preflight_check(lookup, concepts, [])  # empty ETCSL
    verdict = report["concepts"][0]
    assert verdict["etcsl_passages"] == 0
    assert "zero_etcsl_passages" in verdict["failure_reasons"]
    assert verdict["status"] == "fail"


def test_report_schema_stable():
    from scripts.analysis.preflight_concept_check import preflight_check

    lookup = _preflight_stub()
    report = preflight_check(lookup, [], [])
    assert "preflight_schema_version" in report
    assert report["preflight_schema_version"] == 1
    assert "concepts" in report
    assert "preflight_date" in report
```

- [ ] **Step 3: Run tests, verify they fail**

```bash
pytest tests/analysis/test_preflight_concept_check.py -v
```
Expected: 5 FAIL on ImportError.

- [ ] **Step 4: Implement `scripts/analysis/preflight_concept_check.py`**

```python
"""
Pre-flight check for the Sumerian cosmogony document's concept slate.

For each candidate concept, reports:
  - Is the Sumerian token in the fused vocab?
  - Is the English seed in Gemma and GloVe vocabs?
  - Does find_both return non-degenerate top-5 matches?
  - How many ETCSL passages contain the token?

Produces a status (pass | fail) per concept, with failure reasons and
warnings. Output JSON is consulted before the generators and document
prose are produced.

See: docs/superpowers/specs/2026-04-19-sumerian-cosmogony-document-design.md
"""
from __future__ import annotations

import datetime as _dt

PREFLIGHT_SCHEMA_VERSION = 1
DEGENERATE_LEN_THRESHOLD = 2  # tokens of length <= 2 are flagged as degenerate
DEGENERATE_TOP5_MIN_FRACTION = 0.4  # <=40% multi-char tokens in top 5 -> flagged


def preflight_check(
    lookup,
    candidate_concepts: list[dict],
    etcsl_texts: list[dict],
) -> dict:
    """Validate each concept against the current lookup + ETCSL corpus."""
    verdicts = []
    for concept in candidate_concepts:
        sum_tok = concept["sumerian"]
        eng_seed = concept["english"]

        sum_in_vocab = sum_tok in lookup.vocab

        # English-vocab check via find() returning non-empty.
        eng_in_gemma = bool(lookup.find(eng_seed, top_k=1, space="gemma"))
        eng_in_glove = bool(lookup.find(eng_seed, top_k=1, space="glove"))

        # Top-5 quality check.
        top5 = lookup.find_both(eng_seed, top_k=5) if eng_in_gemma or eng_in_glove else {"gemma": [], "glove": []}
        all_top5 = list(top5.get("gemma", [])) + list(top5.get("glove", []))
        if all_top5:
            multi_char_count = sum(1 for w, _ in all_top5 if len(w) > DEGENERATE_LEN_THRESHOLD)
            degenerate_fraction = 1.0 - (multi_char_count / len(all_top5))
        else:
            degenerate_fraction = 1.0

        # ETCSL passage count.
        etcsl_count = 0
        for text in etcsl_texts:
            for line in text.get("lines", []):
                if sum_tok in (line.get("transliteration") or "").split():
                    etcsl_count += 1

        failure_reasons = []
        warnings = []

        if not sum_in_vocab:
            failure_reasons.append("sumerian_vocab_miss")
        if not eng_in_gemma and not eng_in_glove:
            failure_reasons.append("english_missing_both_spaces")
        if etcsl_count == 0:
            failure_reasons.append("zero_etcsl_passages")
        if degenerate_fraction > 0.5:
            warnings.append("degenerate_top5")
        if not eng_in_gemma:
            warnings.append("english_missing_gemma")
        if not eng_in_glove:
            warnings.append("english_missing_glove")

        verdicts.append({
            "concept": concept,
            "status": "fail" if failure_reasons else "pass",
            "sumerian_in_vocab": sum_in_vocab,
            "english_in_gemma": eng_in_gemma,
            "english_in_glove": eng_in_glove,
            "etcsl_passages": etcsl_count,
            "degenerate_fraction_top5": round(degenerate_fraction, 3),
            "failure_reasons": failure_reasons,
            "warnings": warnings,
            "top5_gemma": top5.get("gemma", []),
            "top5_glove": top5.get("glove", []),
        })

    return {
        "preflight_schema_version": PREFLIGHT_SCHEMA_VERSION,
        "preflight_date": _dt.date.today().isoformat(),
        "concepts": verdicts,
    }
```

- [ ] **Step 5: Run tests, verify all pass**

```bash
pytest tests/analysis/test_preflight_concept_check.py -v
```
Expected: 5 pass.

- [ ] **Step 6: Full suite regression**

```bash
pytest tests/ --ignore=tests/test_integration.py -q
```
Expected: 142 pass.

- [ ] **Step 7: Commit the module + concept slate**

```bash
cd /Users/crashy/Development/cuneiformy
git add scripts/analysis/cosmogony_concepts.py \
        scripts/analysis/preflight_concept_check.py \
        tests/analysis/test_preflight_concept_check.py
git commit -m "feat: add cosmogony concept slate + preflight check module"
```

- [ ] **Step 8: Run pre-flight on real data**

Create a small driver script inline (do not commit; this is a one-off):

```bash
cd /Users/crashy/Development/cuneiformy
python3 <<'EOF'
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))

from final_output.sumerian_lookup import SumerianLookup
from scripts.analysis.cosmogony_concepts import PRIMARY_CONCEPTS, ALTERNATE_CONCEPTS
from scripts.analysis.preflight_concept_check import preflight_check

ROOT = Path(".")
print("Loading GloVe (~1 min)...")
glove_vocab = []
glove_vectors = []
with open(ROOT / "data/processed/glove.6B.300d.txt", encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split(" ")
        glove_vocab.append(parts[0])
        glove_vectors.append([float(x) for x in parts[1:]])
import numpy as np
glove_vectors = np.array(glove_vectors, dtype=np.float32)

lookup = SumerianLookup(
    gemma_vectors_path=str(ROOT / "final_output/sumerian_aligned_gemma_vectors.npz"),
    glove_vectors_path=str(ROOT / "final_output/sumerian_aligned_vectors.npz"),
    vocab_path=str(ROOT / "final_output/sumerian_aligned_vocab.pkl"),
    gemma_english_path=str(ROOT / "models/english_gemma_whitened_768d.npz"),
    glove_english_vectors=glove_vectors,
    glove_english_vocab=glove_vocab,
)

with open(ROOT / "data/raw/etcsl_texts.json") as f:
    etcsl_texts = json.load(f)

all_candidates = PRIMARY_CONCEPTS + ALTERNATE_CONCEPTS
report = preflight_check(lookup, all_candidates, etcsl_texts)

out_path = ROOT / f"results/cosmogony_preflight_{date.today().isoformat()}.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"Report: {out_path}")

# Summary line per concept
for v in report["concepts"]:
    tag = v["concept"]["sumerian"]
    print(f"  {tag:>10s}: status={v['status']:<4s}  in_vocab={v['sumerian_in_vocab']}  etcsl={v['etcsl_passages']:>4d}  deg_frac={v['degenerate_fraction_top5']:.2f}")
EOF
```

Runtime: ~1 minute (GloVe load). Capture the stdout summary table — this tells you which concepts pass, which need substitution, and which warnings fired.

- [ ] **Step 9: Evaluate the pre-flight result and lock the concept slate**

Review the `results/cosmogony_preflight_<today>.json` output:
- For each of the 5 primary concepts, verify `status: "pass"`.
- If ANY primary concept has `status: "fail"`, substitute from `ALTERNATE_CONCEPTS` following the substitution rules in the spec:
  - `sumerian_vocab_miss` or `zero_etcsl_passages` → substitute (pick an alternate with matching theme if possible).
  - `degenerate_top5` (warning, not failure) → flag in §12 appendix but do NOT auto-substitute — this is an informative finding about anchor quality.
- Update `scripts/analysis/cosmogony_concepts.py`'s `PRIMARY_CONCEPTS` list to reflect any substitutions.
- Re-run pre-flight to confirm the final slate all passes.

Document the substitution reasoning (if any) as a comment above `PRIMARY_CONCEPTS` in the config file — this feeds into §12 appendix during prose writing.

- [ ] **Step 10: Commit pre-flight report + any slate updates**

```bash
cd /Users/crashy/Development/cuneiformy
TODAY=$(date +%Y-%m-%d)
git add -f results/cosmogony_preflight_${TODAY}.json
git add scripts/analysis/cosmogony_concepts.py  # if updated
git commit -m "chore: commit cosmogony preflight report + finalize concept slate"
```

---

## Task 5: Top-level entry points (TDD for determinism)

**Files:**
- Create: `scripts/analysis/generate_cosmogony_tables.py`
- Create: `scripts/analysis/generate_cosmogony_figures.py`
- Create: `docs/cosmogony_tables.json`
- Create: `docs/figures/cosmogony/*.png` (7 files)

### Setup note

These two scripts are the delivery drivers. They consume the 4 analysis modules (semantic_field, english_displacement, etcsl_passage_finder, umap_projection) plus the locked concept slate and the real SumerianLookup, and produce the document's data inputs deterministically.

- [ ] **Step 1: Write `generate_cosmogony_tables.py`**

```python
"""
Generate cosmogony_tables.json from the final concept slate.

For each concept, compute:
  - Top-10 nearest Sumerian neighbors in both spaces (dual-view)
  - 2-3 analogy probe results
  - English displacement numbers in both spaces
  - 1-2 ETCSL passage excerpts

All tables committed to docs/cosmogony_tables.json — this is the canonical
source the document prose references.

Usage:
    cd /Users/crashy/Development/cuneiformy
    python scripts/analysis/generate_cosmogony_tables.py

See: docs/superpowers/specs/2026-04-19-sumerian-cosmogony-document-design.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from final_output.sumerian_lookup import SumerianLookup
from scripts.analysis.cosmogony_concepts import PRIMARY_CONCEPTS, ANUNNAKI_VOCABULARY
from scripts.analysis.english_displacement import english_displacement
from scripts.analysis.etcsl_passage_finder import find_passages

ROOT = _ROOT
TABLES_PATH = ROOT / "docs" / "cosmogony_tables.json"

# Analogy probes — curated per concept. Each is (a, b, c, space) such that
# find_analogy(a, b, c, space=space) tests a specific interpretive claim.
ANALOGY_PROBES = {
    "abzu":   [("ocean", "water", "deep", "gemma")],
    "zi":     [("breath", "air", "life", "gemma")],
    "nam":    [("essence", "name", "being", "gemma")],
    "namtar": [("fate", "name", "decree", "gemma")],
    "me":     [("decree", "order", "essence", "gemma")],
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


def _top10_dual_view(lookup, english_seed: str) -> dict:
    both = lookup.find_both(english_seed, top_k=10)
    return {
        "gemma": [{"sumerian": w, "similarity": round(float(s), 4)} for w, s in both["gemma"]],
        "glove": [{"sumerian": w, "similarity": round(float(s), 4)} for w, s in both["glove"]],
    }


def _analogy_probes_for(lookup, concept_tag: str) -> list:
    probes = ANALOGY_PROBES.get(concept_tag, [])
    out = []
    for a, b, c, space in probes:
        result = lookup.find_analogy(a, b, c, top_k=5, space=space)
        out.append({
            "query": f"{a} : {b} :: {c} : ?",
            "space": space,
            "results": [{"sumerian": w, "similarity": round(float(s), 4)} for w, s in result],
        })
    return out


def main() -> int:
    lookup = _load_lookup()

    with open(ROOT / "data/raw/etcsl_texts.json") as f:
        etcsl = json.load(f)

    concepts_out = {}
    for concept in PRIMARY_CONCEPTS:
        sum_tok = concept["sumerian"]
        eng_seed = concept["english"]

        top10 = _top10_dual_view(lookup, eng_seed)

        displacement = {
            "gemma": english_displacement(lookup, sum_tok, eng_seed, space="gemma"),
            "glove": english_displacement(lookup, sum_tok, eng_seed, space="glove"),
        }

        analogies = _analogy_probes_for(lookup, sum_tok)

        passages = find_passages(sum_tok, etcsl, max_passages=2, context_lines=1)

        concepts_out[sum_tok] = {
            "concept": concept,
            "top10_dual_view": top10,
            "english_displacement": displacement,
            "analogy_probes": analogies,
            "etcsl_passages": passages,
        }

    tables = {
        "schema_version": 1,
        "concept_slate": PRIMARY_CONCEPTS,
        "anunnaki_vocabulary": ANUNNAKI_VOCABULARY,
        "concepts": concepts_out,
    }

    TABLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TABLES_PATH, "w") as f:
        json.dump(tables, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote: {TABLES_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Write `generate_cosmogony_figures.py`**

```python
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

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from final_output.sumerian_lookup import SumerianLookup
from scripts.analysis.cosmogony_concepts import (
    PRIMARY_CONCEPTS, ANUNNAKI_VOCABULARY, COSMOGONIC_POLES,
)
from scripts.analysis.semantic_field import (
    compute_pairwise_distances, render_semantic_field_heatmap,
)
from scripts.analysis.umap_projection import umap_cosmogonic_vocabulary

ROOT = _ROOT
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
```

- [ ] **Step 3: Run both generators**

```bash
cd /Users/crashy/Development/cuneiformy
python scripts/analysis/generate_cosmogony_tables.py
python scripts/analysis/generate_cosmogony_figures.py
```

Expected: `docs/cosmogony_tables.json` and 7 PNG files under `docs/figures/cosmogony/` are produced.

- [ ] **Step 4: Determinism check**

Run the generators twice, diff the outputs:

```bash
cp docs/cosmogony_tables.json /tmp/tables_first.json
python scripts/analysis/generate_cosmogony_tables.py
diff /tmp/tables_first.json docs/cosmogony_tables.json
```
Expected: empty diff. If any differences appear, there's non-determinism to fix.

Figures: matplotlib rendering can vary by matplotlib version; byte-identical PNG determinism is not a guaranteed property. We accept this — the underlying coords are deterministic via `_compute_embedding` seeded UMAP. Manual inspection of figures suffices.

- [ ] **Step 5: Full suite regression**

```bash
pytest tests/ --ignore=tests/test_integration.py -q
```
Expected: 142 pass (no new tests in this task — the entry points are orchestrators; their correctness is tested via the underlying module tests).

- [ ] **Step 6: Commit scripts + generated tables**

```bash
cd /Users/crashy/Development/cuneiformy
git add scripts/analysis/generate_cosmogony_tables.py \
        scripts/analysis/generate_cosmogony_figures.py \
        docs/cosmogony_tables.json
git commit -m "feat: add cosmogony tables + figures generators"
```

- [ ] **Step 7: Commit generated figures**

```bash
cd /Users/crashy/Development/cuneiformy
git add docs/figures/cosmogony/
git commit -m "chore: commit generated cosmogony figures"
```

---

## Task 6: Write the document prose

**Files:**
- Create: `docs/sumerian_cosmogony.md`

### Setup note

This is the longest task. ~10,500 words of carefully-hedged prose grounded in the already-generated tables (`docs/cosmogony_tables.json`) and figures (`docs/figures/cosmogony/*.png`). No code tests, but clear success criteria.

Write sections in order (§0 → §12). For each section, draft the prose referencing the generated data, then review: does every numeric claim trace to a specific JSON path or figure? Is every Sumerological claim backed by an ETCSL passage (from `etcsl_passages` in the tables JSON) or a named secondary source?

### Writing discipline

For each deep dive (§4-8), the 8-section template is:
1. Anchor reading — `concepts.<tag>.concept` + Sumerological context from references.
2. Nearest Sumerian neighbors — `concepts.<tag>.top10_dual_view.gemma` top 10 in a table, commentary on what surfaces.
3. Semantic-field map — reference the generated heatmap file.
4. Dual-view divergence — compare `top10_dual_view.gemma` vs `top10_dual_view.glove`, call out agreements and disagreements.
5. Analogy probes — `concepts.<tag>.analogy_probes` results in a table, commentary.
6. English displacement — one-number callout from `concepts.<tag>.english_displacement.gemma.cosine_similarity`.
7. Source-text grounding — quote from `concepts.<tag>.etcsl_passages`, contextualize.
8. Interpretive synthesis — one hedged cosmogonic claim with alternatives noted.

### References to curate (§11)

- ETCSL text IDs actually cited in §7 passages (from `etcsl_passages[*].text_id`).
- Thorkild Jacobsen — *The Treasures of Darkness* (key secondary source for Sumerian theology).
- Samuel Noah Kramer — *Sumerian Mythology* and *History Begins at Sumer*.
- Jeremy Black, Graham Cunningham, Eleanor Robson, Gábor Zólyomi — *The Literature of Ancient Sumer* (Oxford, 2004).
- Daniel Foxvog — *Introduction to Sumerian Grammar* (especially useful for explaining the `nam-` essence prefix in §6).
- Piotr Michalowski — writings on ME and Sumerian cosmology.

Pull real citation data (publication year, publisher) during writing; do NOT fabricate.

- [ ] **Step 1: Draft §0 Abstract + §1 Introduction + §2 Methodology**

Target: ~1,450 words total across these three sections.

Abstract: what the document does, who it's for, what it finds at the highest level.

Introduction: what is geometric translation, why Sumerian cosmogony as the case study, the 5 concepts we'll deep-dive, a one-sentence preview of each finding.

Methodology: describe the Cuneiformy pipeline (half a page), the SumerianLookup API, the per-concept 8-section template, and the caveats — alignment quality (cite the 52.13% top-1), anchor bias (cite Workstream 2b-pre diagnostic findings on ePSD2 citation-form normalization), corpus sparsity, and the note that "geometric translation" makes no claim about what Sumerians believed, only about what the embedding geometry is consistent with.

- [ ] **Step 2: Draft §3 The Cosmogonic Arc (narrative spine)**

Target: ~1,500 words + reference to `docs/figures/cosmogony/anunnaki_narrative_umap.png`.

Structure: primordial waters (Nammu) → An/Ki separation → the Anunnaki form and take up their functions → Enki's creation of humans from clay with breath → Anunnaki decree destinies → `me` distributed to cities (especially Inanna's theft from Enki). End with a sentence-long pointer: "The next five sections zoom into five pivotal terms at five pivotal moments."

Cite ETCSL texts that narrate each phase (e.g., t.1.1.1 for Eridu Genesis, t.1.1.2 for Enki and Ninmah, etc., verified against actual ETCSL IDs during writing).

- [ ] **Step 3: Draft §4 abzu**

Target: ~1,600 words.

Use the 8-section template. Pull:
- Top-10 gemma neighbors from `concepts.abzu.top10_dual_view.gemma`.
- Semantic-field heatmap reference: `docs/figures/cosmogony/abzu_semantic_field_heatmap.png`.
- Gemma vs GloVe dual-view from both sides of `top10_dual_view`.
- Analogy probes from `concepts.abzu.analogy_probes`.
- English displacement from `concepts.abzu.english_displacement.gemma.cosine_similarity`.
- 1-2 ETCSL passages from `concepts.abzu.etcsl_passages`.

Interpretive synthesis: what does the geometric neighborhood of `abzu` suggest about pre-creation Sumerian cosmology? How does it compare to English "deep" or "ocean"? Hedge: this is embedding geometry consistent with X, not proof that X was Sumerian belief.

- [ ] **Step 4: Draft §5 zi (breath)**

Same pattern. ~1,600 words.

- [ ] **Step 5: Draft §6 nam (essence)**

Same pattern. Note: `nam-` is a grammatical prefix in Sumerian; reference Foxvog's grammar for the essence-nominalizer explanation. The geometric finding will probably show `nam` clustering with other essence-creating prefixes and deep conceptual terms. ~1,600 words.

- [ ] **Step 6: Draft §7 namtar (fate)**

Same pattern. This is the concept most tightly tied to the Anunnaki's cosmogonic function (they decree destinies). The geometric finding will likely show `namtar` clustering with cutting (`tar`), naming (`mu`), and essence (`nam`). Interpretive synthesis should address the claim that Sumerian fate is "spoken + cut + named" rather than "what happens to you." ~1,600 words.

- [ ] **Step 7: Draft §8 me (divine decrees)**

Same pattern. `me` is the most genuinely-alien concept in the slate. Interpretive synthesis should name what's different between Sumerian `me` and any English approximation (decree, essence, office). ~1,600 words.

- [ ] **Step 8: Draft §9 Synthesis**

Target: ~1,500 words + reference to `docs/figures/cosmogony/cosmogony_axis_projection.png`.

Structure:
- Recap: what the five deep dives collectively found.
- The cosmogonic axis figure: show where each of the 5 concepts projects on the primordial→decree axis. Comment on surprises (e.g., if `nam` sits closer to the decree pole than to primordial, that says something about Sumerian cosmogonic grammar).
- Connection to `RESEARCH_VISION.md` losses/gains thesis: which of these concepts have partial English analogs, which are geometrically displaced, which have no English counterpart at all.
- Explicit limits: what geometric translation does NOT tell us. The method is consistent with multiple interpretations; it narrows the space of plausible readings but doesn't adjudicate among them. Sumerological ground truth remains the arbiter.

- [ ] **Step 9: Draft §10 Reproducibility + §11 References + §12 Appendix**

§10: ~250 words on how to regenerate the tables, figures, and prose. Name the pinned git commit (the commit after Task 5).

§11: curated reference list. Only citations actually used in prose. ETCSL text IDs with brief descriptions.

§12: report from the pre-flight appendix. Show the pre-flight JSON findings, which concepts were substituted (if any), and why.

- [ ] **Step 10: Internal consistency check**

Before committing, re-read the document and verify:

- Every numeric claim traces to a specific table row or figure. Draft a checklist: for each number in the prose, the source path in `cosmogony_tables.json` or figure filename.
- Every Sumerological claim has a citation: either an ETCSL text ID or a named secondary source. No unreferenced claims.
- No TODOs, no `[...]` placeholders.
- No claim of what Sumerians "really believed" — all interpretive claims are hedged as "the geometry is consistent with…"
- Deep-dive ordering matches the narrative-spine arc: abzu → zi → nam → namtar → me.

- [ ] **Step 11: Commit**

```bash
cd /Users/crashy/Development/cuneiformy
git add docs/sumerian_cosmogony.md
git commit -m "docs: add Sumerian cosmogony document (10,500 words, 5 deep dives)"
```

---

## Task 7: Journal entry + README refresh + final sanity

**Files:**
- Modify: `docs/EXPERIMENT_JOURNAL.md`
- Modify: `README.md`

### Setup note

Delivery step. Ships the document to the repo's top-level research surfaces so readers find it.

- [ ] **Step 1: Add journal entry**

Insert in `docs/EXPERIMENT_JOURNAL.md` AFTER the preamble and BEFORE the existing Workstream 2b entry:

```markdown
## <TODAY> — Sumerian Cosmogony Document shipped

**Hypothesis:** With the Workstream 2b alignment at 52.13% top-1, the embedding geometry is rich enough to support methodology-driven interpretation of Sumerian cosmogonic vocabulary. Writing a case study on the Anunnaki cosmogonic cycle exercises the dual-view SumerianLookup API, surfaces whether the geometry says anything non-obvious, and produces a shareable research artifact.

**Method:** New `scripts/analysis/` directory with 5 focused modules (semantic_field, english_displacement, etcsl_passage_finder, umap_projection, preflight_concept_check) + 2 top-level generators. Pre-flight validated 5 concepts (`abzu`, `zi`, `nam`, `namtar`, `me`) against vocab + ETCSL + top-5 quality. Generators produced `docs/cosmogony_tables.json` and 7 committed figures deterministically. Prose written from the committed data — no hand-calculated numbers, no uncited Sumerological claims.

**Result:** `docs/sumerian_cosmogony.md` (~10,500 words, 15-18 printed pages). Five concept deep dives following an 8-section paper-grade template. [1-2 sentences of headline findings — fill in during Task 7 Step 1 based on the actual prose].

**Takeaway:** [Fill in 2-3 sentences after the document exists: what did the geometry actually show? What are the strongest claims? What are the most significant limits?]

**Artifacts / commits:** `docs/sumerian_cosmogony.md`, `docs/cosmogony_tables.json`, `docs/figures/cosmogony/*.png`, `scripts/analysis/*.py`, `tests/analysis/*.py`, `results/cosmogony_preflight_<DATE>.json`. Spec: `docs/superpowers/specs/2026-04-19-sumerian-cosmogony-document-design.md`. Plan: `docs/superpowers/plans/2026-04-19-sumerian-cosmogony-document.md`.
```

The three `[...]` placeholders are filled in after the prose exists; they describe the actual findings, not hypothesized ones.

- [ ] **Step 2: Update README.md**

Add to the Research Progress section in `README.md`:

```markdown
- **<TODAY> — Sumerian Cosmogony document:** A methodology-driven ~10,500-word case study on the Anunnaki cosmogonic cycle, using the 52%-top-1 whitened-Gemma alignment for geometric translation of five pivotal terms (`abzu`, `zi`, `nam`, `namtar`, `me`). See [`docs/sumerian_cosmogony.md`](docs/sumerian_cosmogony.md).
```

Insert as the NEW FIRST bullet under "Recent findings (newest first):" in the Research Progress section.

- [ ] **Step 3: Final sanity run**

```bash
cd /Users/crashy/Development/cuneiformy
pytest tests/ --ignore=tests/test_integration.py -q
```
Expected: 142 passed, 0 failed. No regressions from any prior workstream.

Verify all success criteria from the spec:
- [ ] All 5 deep dives have all 8 template sub-sections; no TODOs.
- [ ] All 7 figures exist at committed PNG paths.
- [ ] `docs/cosmogony_tables.json` regenerates byte-identically.
- [ ] Every numeric claim traces to tables JSON / figures.
- [ ] Every Sumerological claim cites ETCSL or named secondary.
- [ ] Pre-flight report appendix included; substitutions documented.
- [ ] §10 names the pinned git commit.

- [ ] **Step 4: Commit journal + README**

```bash
cd /Users/crashy/Development/cuneiformy
git add docs/EXPERIMENT_JOURNAL.md README.md
git commit -m "docs: journal cosmogony document + link from README"
```

---

## Self-Review

Spec requirements matched to tasks:

- **`docs/sumerian_cosmogony.md` with 13 sections + 8-section per-concept template** → Task 6.
- **`scripts/analysis/` 5 focused modules** → Tasks 1 (semantic_field), 2 (english_displacement + etcsl_passage_finder), 3 (umap_projection), 4 (preflight_concept_check).
- **2 top-level generators** → Task 5.
- **7 committed PNG figures** → Task 5 Step 7.
- **`docs/cosmogony_tables.json`** → Task 5 Step 6.
- **`results/cosmogony_preflight_<DATE>.json`** → Task 4 Step 10.
- **Test coverage for all analysis modules** → Tasks 1 (5 tests), 2 (9 tests), 3 (3 tests), 4 (5 tests). 22 new tests total.
- **Pre-flight concept availability check with substitution rules** → Task 4.
- **Deterministic regeneration** → Task 5 Step 4 checks `cosmogony_tables.json` determinism; UMAP coord-determinism tested in Task 3.
- **Dependencies pinned (`umap-learn`, `matplotlib`)** → Task 3 Step 1.
- **Journal entry + README link** → Task 7.

Placeholder scan:
- Task 4 config file has "Substitutions drawn on if a primary concept fails" — this is an intentional branch point, not a placeholder. The substitution rules are explicitly specified.
- Task 6 Steps 3-7 reference "commentary" and "interpretive synthesis" without verbatim prose — these are hand-writing tasks where the prose is produced against real generated data. The 8-section template is explicit; the interpretive content is inherent to the section.
- Task 7 journal entry template has 3 explicit `[...]` placeholders flagged as "fill in after the prose exists." Correct — those values depend on what the generators actually produced.
- No `TBD`, `TODO`, "similar to Task N", or "handle edge cases" patterns.

Type consistency:
- `SumerianLookup` API used consistently across Tasks 1-5 (`find`, `find_both`, `find_analogy`, `._spaces[space]["sum_norm"]`, `.vocab`, `._spaces[space]["eng_norm"]`, `._spaces[space]["eng_vocab_map"]`).
- Module function signatures in tests match implementations.
- Concept slate `{"sumerian": str, "english": str, "theme": str}` dict shape consistent across `cosmogony_concepts.py`, pre-flight tests, and generators.
- Table JSON schema consistent between `generate_cosmogony_tables.py` output and the document's Task 6 reading conventions.
- Figure filenames consistent across Task 5 generator and Task 6 prose references.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-19-sumerian-cosmogony-document.md`. Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task with two-stage review (spec compliance + code quality). Tasks 1-5 are code; Task 6 is prose (different discipline — may want controller-in-the-loop per section). Task 7 is delivery.

**2. Inline Execution** — batch execution via `superpowers:executing-plans` with checkpoints.

Which approach?
