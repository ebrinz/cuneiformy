# Geometric Narrative Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a three-stage pipeline that performs geometric comparison of Sumerian vs English embedding spaces, generates "geometric translations" of ETCSL passages, and produces a long-form philosophical essay as a compilable LaTeX document.

**Architecture:** Three scripts run sequentially — `geometric_compare.py` produces findings + figures, `geometric_translate.py` extracts parallel passages with geometric glosses, `generate_narrative.py` assembles everything into a LaTeX essay. Each script reads the previous script's JSON output.

**Tech Stack:** numpy, scipy, scikit-learn (already installed); umap-learn, matplotlib, seaborn (to add); LaTeX via pdflatex for PDF compilation.

---

## File Structure

```
scripts/
├── geometric_compare.py      # Stage 1: geometric analysis of 3 concept domains
├── geometric_translate.py     # Stage 2: parallel passage extraction + geometric glosses
└── generate_narrative.py      # Stage 3: LaTeX document assembly

tests/
├── test_geometric_compare.py
├── test_geometric_translate.py
├── test_generate_narrative.py
└── test_narrative_integration.py

results/                       # Created at runtime
├── geometric_findings.json
├── parallel_passages.json
└── figures/
    ├── creation_distance_diff.png
    ├── creation_umap.png
    ├── fate_distance_diff.png
    ├── fate_umap.png
    ├── self_distance_diff.png
    ├── self_umap.png
    └── domain_comparison.png

output/                        # Created at runtime
├── geometric_narrative.tex
├── geometric_narrative.pdf
└── references.bib
```

---

### Task 1: Add Dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add umap-learn, matplotlib, seaborn to requirements.txt**

```
umap-learn>=0.5.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

Append these three lines to the existing `requirements.txt`.

- [ ] **Step 2: Install**

Run: `pip install umap-learn matplotlib seaborn`

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: add umap-learn, matplotlib, seaborn for geometric analysis"
```

---

### Task 2: Vector Loading Helpers + Tests

**Files:**
- Create: `scripts/geometric_compare.py`
- Create: `tests/test_geometric_compare.py`

This task builds the data loading and core geometric primitives that all subsequent analysis depends on.

- [ ] **Step 1: Write failing tests for vector loading and cosine distance utilities**

Create `tests/test_geometric_compare.py`:

```python
import numpy as np
import pytest


def test_load_sumerian_vectors():
    """Load aligned Sumerian vectors and vocab from production output."""
    from scripts.geometric_compare import load_sumerian_vectors

    vectors, vocab, word_to_idx = load_sumerian_vectors(
        "final_output/sumerian_aligned_vectors.npz",
        "final_output/sumerian_aligned_vocab.pkl",
    )
    assert vectors.shape[0] == len(vocab)
    assert vectors.shape[1] == 300
    assert isinstance(word_to_idx, dict)
    assert word_to_idx[vocab[0]] == 0


def test_load_glove_vectors():
    """Load GloVe vectors and build vocab index."""
    from scripts.geometric_compare import load_glove_vectors

    vectors, vocab, word_to_idx = load_glove_vectors(
        "data/processed/glove.6B.300d.txt", max_words=1000
    )
    assert vectors.shape == (1000, 300)
    assert len(vocab) == 1000
    assert "the" in word_to_idx


def test_cosine_distance_matrix():
    """Pairwise cosine distance matrix for a set of vectors."""
    from scripts.geometric_compare import cosine_distance_matrix

    vecs = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
    D = cosine_distance_matrix(vecs)
    assert D.shape == (3, 3)
    assert abs(D[0, 0]) < 1e-6  # self-distance is 0
    assert abs(D[0, 1] - 1.0) < 1e-6  # orthogonal = distance 1
    assert D[0, 2] < D[0, 1]  # [1,1,0] closer to [1,0,0] than [0,1,0] is


def test_nearest_neighbors():
    """Find k nearest neighbors by cosine similarity."""
    from scripts.geometric_compare import nearest_neighbors

    vecs = np.array([[1, 0], [0.9, 0.1], [0, 1], [-1, 0]], dtype=np.float32)
    vocab = ["a", "b", "c", "d"]
    query = np.array([1, 0], dtype=np.float32)
    results = nearest_neighbors(query, vecs, vocab, k=2)
    assert results[0][0] == "a"
    assert results[1][0] == "b"
    assert len(results) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_geometric_compare.py -v`
Expected: FAIL — `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Implement vector loading and utilities**

Create `scripts/geometric_compare.py`:

```python
"""
Geometric comparison of Sumerian and English embedding spaces.

Analyzes three conceptual domains (creation, fate, self) to find
where Sumerian semantic geometry diverges from English.
"""
import json
import os
import pickle

import numpy as np
from scipy.spatial.distance import cosine


def load_sumerian_vectors(vectors_path, vocab_path):
    """Load aligned Sumerian vectors and vocab."""
    data = np.load(vectors_path, allow_pickle=True)
    vectors = data["vectors"].astype(np.float32)
    with open(vocab_path, "rb") as f:
        vocab = list(pickle.load(f))
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    return vectors, vocab, word_to_idx


def load_glove_vectors(glove_path, max_words=None):
    """Load GloVe vectors from text file."""
    vectors = []
    vocab = []
    with open(glove_path, "r") as f:
        for i, line in enumerate(f):
            if max_words and i >= max_words:
                break
            parts = line.strip().split(" ")
            vocab.append(parts[0])
            vectors.append([float(x) for x in parts[1:]])
    vectors = np.array(vectors, dtype=np.float32)
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    return vectors, vocab, word_to_idx


def cosine_distance_matrix(vectors):
    """Pairwise cosine distance matrix. 0 = identical, 2 = opposite."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normed = vectors / norms
    sims = normed @ normed.T
    return 1 - sims


def nearest_neighbors(query, vectors, vocab, k=10):
    """Find k nearest neighbors to query vector by cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normed = vectors / norms
    q_norm = query / (np.linalg.norm(query) + 1e-10)
    sims = normed @ q_norm
    top_idx = np.argsort(sims)[::-1][:k]
    return [(vocab[i], float(sims[i])) for i in top_idx]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_geometric_compare.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/geometric_compare.py tests/test_geometric_compare.py
git commit -m "feat: add vector loading and geometric primitives"
```

---

### Task 3: Domain Analysis Engine + Tests

**Files:**
- Modify: `scripts/geometric_compare.py`
- Modify: `tests/test_geometric_compare.py`

Builds the per-domain analysis functions: cluster extraction, distance matrix diff, neighborhood divergence, centroid displacement, cluster shape, and merge/split detection.

- [ ] **Step 1: Write failing tests for domain analysis functions**

Append to `tests/test_geometric_compare.py`:

```python
def test_cluster_extraction():
    """Extract Sumerian cluster for a set of English seed words."""
    from scripts.geometric_compare import extract_cluster

    # 5 Sumerian words in 3d space, 4 English words in 3d space
    sum_vecs = np.random.randn(5, 3).astype(np.float32)
    sum_vocab = ["lugal", "e2", "dingir", "nam", "an"]
    sum_w2i = {w: i for i, w in enumerate(sum_vocab)}

    eng_vecs = np.random.randn(4, 3).astype(np.float32)
    eng_vocab = ["king", "house", "god", "fate"]
    eng_w2i = {w: i for i, w in enumerate(eng_vocab)}

    seeds = ["king", "house"]
    cluster = extract_cluster(
        seeds, sum_vecs, sum_vocab, sum_w2i, eng_vecs, eng_vocab, eng_w2i, top_k=2
    )
    assert "seeds" in cluster
    assert "sumerian_words" in cluster
    assert "discovered_english" in cluster
    assert len(cluster["sumerian_words"]) > 0


def test_distance_matrix_diff():
    """Compute difference between English-native and Sumerian-projected distance matrices."""
    from scripts.geometric_compare import distance_matrix_diff

    eng_vecs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    sum_proxy_vecs = np.array([[1, 0, 0], [0.5, 0.5, 0], [0, 0, 1]], dtype=np.float32)
    labels = ["a", "b", "c"]

    diff, eng_dist, sum_dist = distance_matrix_diff(eng_vecs, sum_proxy_vecs, labels)
    assert diff.shape == (3, 3)
    # b moved closer to a in Sumerian space, so diff[0,1] should be negative
    assert diff[0, 1] < 0


def test_neighborhood_divergence():
    """Jaccard divergence between English and Sumerian neighborhoods."""
    from scripts.geometric_compare import neighborhood_divergence

    eng_vecs = np.eye(5, dtype=np.float32)
    sum_vecs = np.eye(5, dtype=np.float32)
    sum_vecs[1] = eng_vecs[0] * 0.9 + eng_vecs[1] * 0.1  # shift word 1 near word 0
    vocab = ["a", "b", "c", "d", "e"]
    seeds = ["a", "b"]

    divs = neighborhood_divergence(seeds, eng_vecs, sum_vecs, vocab, k=3)
    assert len(divs) == 2
    for seed, jaccard in divs.items():
        assert 0 <= jaccard <= 1


def test_centroid_displacement():
    """Compute displacement vector between English and Sumerian cluster centroids."""
    from scripts.geometric_compare import centroid_displacement

    eng_vecs = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    sum_vecs = np.array([[0, 1, 0], [0, 0, 1]], dtype=np.float32)

    displacement = centroid_displacement(eng_vecs, sum_vecs)
    assert displacement.shape == (3,)
    assert np.linalg.norm(displacement) > 0


def test_cluster_shape():
    """Eigenvalue decomposition for cluster shape comparison."""
    from scripts.geometric_compare import cluster_shape

    # Elongated cluster along axis 0
    vecs = np.array([[i, 0, 0] for i in range(10)], dtype=np.float32)
    eigenvalues, effective_dim = cluster_shape(vecs)
    assert len(eigenvalues) == 3
    assert eigenvalues[0] > eigenvalues[1]  # first eigenvalue dominates
    assert effective_dim >= 1


def test_detect_merges_splits():
    """Detect concept merges and splits between English and Sumerian."""
    from scripts.geometric_compare import detect_merges_splits

    eng_dists = np.array([[0, 0.9, 0.1], [0.9, 0, 0.8], [0.1, 0.8, 0]], dtype=np.float32)
    sum_dists = np.array([[0, 0.3, 0.1], [0.3, 0, 0.8], [0.1, 0.8, 0]], dtype=np.float32)
    labels = ["a", "b", "c"]

    merges, splits = detect_merges_splits(eng_dists, sum_dists, labels, threshold=2.0)
    # a and b: eng 0.9, sum 0.3 -> ratio 3.0 -> merge
    assert len(merges) >= 1
    assert merges[0]["pair"] == ("a", "b")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_geometric_compare.py::test_cluster_extraction -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement domain analysis functions**

Append to `scripts/geometric_compare.py`:

```python
def extract_cluster(seeds, sum_vecs, sum_vocab, sum_w2i, eng_vecs, eng_vocab, eng_w2i, top_k=5):
    """Extract expanded cluster: seed words -> Sumerian neighbors -> reverse English."""
    sum_norms = np.linalg.norm(sum_vecs, axis=1, keepdims=True)
    sum_norms[sum_norms == 0] = 1
    sum_normed = sum_vecs / sum_norms

    eng_norms = np.linalg.norm(eng_vecs, axis=1, keepdims=True)
    eng_norms[eng_norms == 0] = 1
    eng_normed = eng_vecs / eng_norms

    valid_seeds = [s for s in seeds if s in eng_w2i]
    sumerian_words = {}
    discovered_english = set()

    for seed in valid_seeds:
        eng_vec = eng_normed[eng_w2i[seed]]
        sims = sum_normed @ eng_vec
        top_idx = np.argsort(sims)[::-1][:top_k]
        for idx in top_idx:
            word = sum_vocab[idx]
            sim = float(sims[idx])
            if sim > 0.1:
                sumerian_words[word] = {"seed": seed, "similarity": sim}
                # Reverse query: what English words is this Sumerian word near?
                rev_sims = eng_normed @ sum_normed[idx]
                rev_top = np.argsort(rev_sims)[::-1][:top_k]
                for ri in rev_top:
                    discovered_english.add(eng_vocab[ri])

    return {
        "seeds": valid_seeds,
        "sumerian_words": sumerian_words,
        "discovered_english": list(discovered_english - set(valid_seeds)),
    }


def distance_matrix_diff(eng_vecs, sum_proxy_vecs, labels):
    """Difference between English-native and Sumerian-projected distance matrices."""
    eng_dist = cosine_distance_matrix(eng_vecs)
    sum_dist = cosine_distance_matrix(sum_proxy_vecs)
    diff = sum_dist - eng_dist  # negative = closer in Sumerian, positive = farther
    return diff, eng_dist, sum_dist


def neighborhood_divergence(seeds, eng_vecs, sum_vecs, vocab, k=20):
    """Jaccard similarity of k-nearest neighbor sets in English vs Sumerian space."""
    w2i = {w: i for i, w in enumerate(vocab)}
    eng_norms = np.linalg.norm(eng_vecs, axis=1, keepdims=True)
    eng_norms[eng_norms == 0] = 1
    eng_normed = eng_vecs / eng_norms
    sum_norms = np.linalg.norm(sum_vecs, axis=1, keepdims=True)
    sum_norms[sum_norms == 0] = 1
    sum_normed = sum_vecs / sum_norms

    divergences = {}
    for seed in seeds:
        if seed not in w2i:
            continue
        idx = w2i[seed]
        eng_sims = eng_normed @ eng_normed[idx]
        sum_sims = sum_normed @ sum_normed[idx]
        eng_neighbors = set(np.argsort(eng_sims)[::-1][1 : k + 1])
        sum_neighbors = set(np.argsort(sum_sims)[::-1][1 : k + 1])
        intersection = eng_neighbors & sum_neighbors
        union = eng_neighbors | sum_neighbors
        divergences[seed] = len(intersection) / len(union) if union else 0
    return divergences


def centroid_displacement(eng_vecs, sum_vecs):
    """Displacement vector from English centroid to Sumerian centroid."""
    eng_centroid = eng_vecs.mean(axis=0)
    sum_centroid = sum_vecs.mean(axis=0)
    return sum_centroid - eng_centroid


def cluster_shape(vectors):
    """Eigenvalue decomposition of covariance matrix. Returns eigenvalues and effective dimensionality."""
    if len(vectors) < 2:
        return np.array([0.0]), 0
    centered = vectors - vectors.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eigenvalues = np.maximum(eigenvalues, 0)
    total = eigenvalues.sum()
    if total == 0:
        return eigenvalues, 0
    normalized = eigenvalues / total
    # Participation ratio as effective dimensionality
    effective_dim = 1.0 / np.sum(normalized**2) if np.sum(normalized**2) > 0 else 0
    return eigenvalues, float(effective_dim)


def detect_merges_splits(eng_dists, sum_dists, labels, threshold=2.0):
    """Find concept pairs where Sumerian and English disagree on distance."""
    n = len(labels)
    merges = []
    splits = []
    for i in range(n):
        for j in range(i + 1, n):
            e = eng_dists[i, j]
            s = sum_dists[i, j]
            if e < 0.01 or s < 0.01:
                continue
            if e / s >= threshold:
                merges.append({
                    "pair": (labels[i], labels[j]),
                    "eng_distance": float(e),
                    "sum_distance": float(s),
                    "ratio": float(e / s),
                })
            elif s / e >= threshold:
                splits.append({
                    "pair": (labels[i], labels[j]),
                    "eng_distance": float(e),
                    "sum_distance": float(s),
                    "ratio": float(s / e),
                })
    merges.sort(key=lambda x: x["ratio"], reverse=True)
    splits.sort(key=lambda x: x["ratio"], reverse=True)
    return merges, splits
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_geometric_compare.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/geometric_compare.py tests/test_geometric_compare.py
git commit -m "feat: add domain analysis engine (clusters, diffs, merges/splits)"
```

---

### Task 4: Visualization + Full Pipeline Runner

**Files:**
- Modify: `scripts/geometric_compare.py`
- Modify: `tests/test_geometric_compare.py`

Adds figure generation (UMAP projections, heatmap diffs) and the `main()` that runs all three domains end-to-end and writes `results/geometric_findings.json` + `results/figures/`.

- [ ] **Step 1: Write failing tests for visualization and pipeline runner**

Append to `tests/test_geometric_compare.py`:

```python
import os


def test_plot_distance_diff(tmp_path):
    """Generate distance diff heatmap PNG."""
    from scripts.geometric_compare import plot_distance_diff

    diff = np.random.randn(5, 5).astype(np.float32)
    labels = ["a", "b", "c", "d", "e"]
    out_path = str(tmp_path / "test_heatmap.png")
    plot_distance_diff(diff, labels, "Test Domain", out_path)
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


def test_plot_umap_overlay(tmp_path):
    """Generate UMAP overlay PNG with English and Sumerian words."""
    from scripts.geometric_compare import plot_umap_overlay

    eng_vecs = np.random.randn(10, 300).astype(np.float32)
    eng_labels = [f"eng_{i}" for i in range(10)]
    sum_vecs = np.random.randn(8, 300).astype(np.float32)
    sum_labels = [f"sum_{i}" for i in range(8)]
    out_path = str(tmp_path / "test_umap.png")
    plot_umap_overlay(eng_vecs, eng_labels, sum_vecs, sum_labels, "Test", out_path)
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


def test_analyze_domain_returns_findings():
    """Full domain analysis returns structured findings dict."""
    from scripts.geometric_compare import analyze_domain

    n_sum, n_eng, dim = 50, 100, 300
    sum_vecs = np.random.randn(n_sum, dim).astype(np.float32)
    sum_vocab = [f"sum{i}" for i in range(n_sum)]
    sum_w2i = {w: i for i, w in enumerate(sum_vocab)}

    eng_vecs = np.random.randn(n_eng, dim).astype(np.float32)
    eng_vocab = ["create", "begin", "birth", "water", "earth"] + [f"eng{i}" for i in range(95)]
    eng_w2i = {w: i for i, w in enumerate(eng_vocab)}

    domain = {
        "name": "Test",
        "seeds": ["create", "begin", "birth", "water", "earth"],
    }

    findings = analyze_domain(
        "test", domain, sum_vecs, sum_vocab, sum_w2i, eng_vecs, eng_vocab, eng_w2i,
        figures_dir=None,
    )
    assert "cluster" in findings
    assert "merges" in findings
    assert "splits" in findings
    assert "centroid_displacement_neighbors" in findings
    assert "neighborhood_divergence" in findings
    assert "cluster_shape_english" in findings
    assert "cluster_shape_sumerian" in findings
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_geometric_compare.py::test_plot_distance_diff -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement visualization functions and analyze_domain**

Append to `scripts/geometric_compare.py`:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


DOMAINS = {
    "creation": {
        "name": "Creation & Origin",
        "seeds": ["create", "begin", "birth", "origin", "emerge", "form",
                  "earth", "water", "heaven", "separate", "divide", "first",
                  "primordial", "chaos", "order"],
        "etcsl_compositions": ["1.1.1", "1.1.2", "1.7.1"],
    },
    "fate": {
        "name": "Fate, Meaning & Purpose",
        "seeds": ["fate", "destiny", "purpose", "decree", "meaning", "life",
                  "death", "name", "order", "tablet", "judge", "decide",
                  "law", "divine"],
        "etcsl_compositions": ["1.1.4", "1.3.1", "4.32.1"],
    },
    "self": {
        "name": "Self, Soul & Consciousness",
        "seeds": ["self", "soul", "spirit", "mind", "heart", "body",
                  "breath", "shadow", "dream", "blood", "eye", "inner",
                  "thought", "will"],
        "etcsl_compositions": ["1.8.1.4", "2.1.1", "1.4.1"],
    },
}


def plot_distance_diff(diff, labels, domain_name, out_path):
    """Save a heatmap of the distance matrix difference."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        diff, xticklabels=labels, yticklabels=labels,
        cmap="RdBu_r", center=0, annot=True, fmt=".2f", ax=ax,
    )
    ax.set_title(f"{domain_name}: Sumerian - English Distance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_umap_overlay(eng_vecs, eng_labels, sum_vecs, sum_labels, domain_name, out_path):
    """UMAP projection with English and Sumerian words overlaid."""
    from umap import UMAP

    all_vecs = np.vstack([eng_vecs, sum_vecs])
    n_eng = len(eng_labels)
    n_neighbors = min(15, len(all_vecs) - 1)
    reducer = UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
    coords = reducer.fit_transform(all_vecs)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(coords[:n_eng, 0], coords[:n_eng, 1], c="#4A90D9", s=60,
               alpha=0.7, label="English", zorder=2)
    ax.scatter(coords[n_eng:, 0], coords[n_eng:, 1], c="#D94A4A", s=60,
               alpha=0.7, label="Sumerian (projected)", zorder=2)
    for i, label in enumerate(eng_labels):
        ax.annotate(label, coords[i], fontsize=8, alpha=0.8, color="#2A5A8A")
    for i, label in enumerate(sum_labels):
        ax.annotate(label, coords[n_eng + i], fontsize=8, alpha=0.8, color="#8A2A2A")
    ax.set_title(f"{domain_name}: Embedding Space Overlay")
    ax.legend()
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def analyze_domain(domain_key, domain, sum_vecs, sum_vocab, sum_w2i,
                   eng_vecs, eng_vocab, eng_w2i, figures_dir=None, top_k=5):
    """Run full geometric analysis for one concept domain."""
    seeds = domain["seeds"]
    valid_seeds = [s for s in seeds if s in eng_w2i]
    if len(valid_seeds) < 2:
        return {"error": f"Only {len(valid_seeds)} valid seeds found"}

    # 1. Cluster extraction
    cluster = extract_cluster(
        valid_seeds, sum_vecs, sum_vocab, sum_w2i, eng_vecs, eng_vocab, eng_w2i, top_k=top_k,
    )

    # 2. Build proxy vectors: for each seed, use the closest Sumerian word's vector
    eng_norms = np.linalg.norm(eng_vecs, axis=1, keepdims=True)
    eng_norms[eng_norms == 0] = 1
    eng_normed = eng_vecs / eng_norms
    sum_norms = np.linalg.norm(sum_vecs, axis=1, keepdims=True)
    sum_norms[sum_norms == 0] = 1
    sum_normed = sum_vecs / sum_norms

    seed_eng_vecs = []
    seed_sum_proxy_vecs = []
    seed_sum_words = []
    for seed in valid_seeds:
        eng_vec = eng_normed[eng_w2i[seed]]
        seed_eng_vecs.append(eng_vec)
        sims = sum_normed @ eng_vec
        best_idx = np.argmax(sims)
        seed_sum_proxy_vecs.append(sum_normed[best_idx])
        seed_sum_words.append(sum_vocab[best_idx])

    seed_eng_vecs = np.array(seed_eng_vecs)
    seed_sum_proxy_vecs = np.array(seed_sum_proxy_vecs)

    # 3. Distance matrix diff
    diff, eng_dist, sum_dist = distance_matrix_diff(seed_eng_vecs, seed_sum_proxy_vecs, valid_seeds)

    # 4. Merge/split detection
    merges, splits = detect_merges_splits(eng_dist, sum_dist, valid_seeds)

    # 5. Neighborhood divergence
    nbr_div = neighborhood_divergence(
        valid_seeds, seed_eng_vecs, seed_sum_proxy_vecs,
        valid_seeds, k=min(5, len(valid_seeds) - 1),
    )

    # 6. Centroid displacement
    disp = centroid_displacement(seed_eng_vecs, seed_sum_proxy_vecs)
    disp_neighbors = nearest_neighbors(disp, eng_vecs, eng_vocab, k=10)

    # 7. Cluster shape
    eng_eigenvalues, eng_eff_dim = cluster_shape(seed_eng_vecs)
    sum_eigenvalues, sum_eff_dim = cluster_shape(seed_sum_proxy_vecs)

    # 8. Figures
    if figures_dir:
        os.makedirs(figures_dir, exist_ok=True)
        plot_distance_diff(
            diff, valid_seeds, domain["name"],
            os.path.join(figures_dir, f"{domain_key}_distance_diff.png"),
        )
        sum_words_in_cluster = list(cluster["sumerian_words"].keys())
        if sum_words_in_cluster:
            sum_cluster_vecs = np.array([sum_normed[sum_w2i[w]] for w in sum_words_in_cluster])
            plot_umap_overlay(
                seed_eng_vecs, valid_seeds,
                sum_cluster_vecs, sum_words_in_cluster,
                domain["name"],
                os.path.join(figures_dir, f"{domain_key}_umap.png"),
            )

    return {
        "domain": domain["name"],
        "valid_seeds": valid_seeds,
        "seed_to_sumerian_proxy": dict(zip(valid_seeds, seed_sum_words)),
        "cluster": {
            "sumerian_words": cluster["sumerian_words"],
            "discovered_english": cluster["discovered_english"],
        },
        "distance_diff": diff.tolist(),
        "merges": merges,
        "splits": splits,
        "neighborhood_divergence": nbr_div,
        "centroid_displacement_neighbors": disp_neighbors,
        "cluster_shape_english": {
            "eigenvalues": eng_eigenvalues.tolist(),
            "effective_dim": eng_eff_dim,
        },
        "cluster_shape_sumerian": {
            "eigenvalues": sum_eigenvalues.tolist(),
            "effective_dim": sum_eff_dim,
        },
    }


def plot_domain_summary(all_findings, out_path):
    """Summary bar chart comparing merge/split counts and effective dimensionality."""
    domains = []
    merge_counts = []
    split_counts = []
    eng_dims = []
    sum_dims = []
    for key, findings in all_findings.items():
        if "error" in findings:
            continue
        domains.append(findings["domain"])
        merge_counts.append(len(findings.get("merges", [])))
        split_counts.append(len(findings.get("splits", [])))
        eng_dims.append(findings["cluster_shape_english"]["effective_dim"])
        sum_dims.append(findings["cluster_shape_sumerian"]["effective_dim"])

    if not domains:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(domains))
    axes[0].bar(x - 0.2, merge_counts, 0.4, label="Merges", color="#D94A4A")
    axes[0].bar(x + 0.2, split_counts, 0.4, label="Splits", color="#4A90D9")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(domains, rotation=15)
    axes[0].set_ylabel("Count")
    axes[0].set_title("Concept Merges & Splits by Domain")
    axes[0].legend()

    axes[1].bar(x - 0.2, eng_dims, 0.4, label="English", color="#4A90D9")
    axes[1].bar(x + 0.2, sum_dims, 0.4, label="Sumerian", color="#D94A4A")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(domains, rotation=15)
    axes[1].set_ylabel("Effective Dimensionality")
    axes[1].set_title("Cluster Shape Complexity")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    """Run geometric comparison across all domains and write results."""
    print("Loading Sumerian vectors...")
    sum_vecs, sum_vocab, sum_w2i = load_sumerian_vectors(
        "final_output/sumerian_aligned_vectors.npz",
        "final_output/sumerian_aligned_vocab.pkl",
    )
    print(f"  {len(sum_vocab)} Sumerian words, {sum_vecs.shape[1]}d")

    print("Loading GloVe vectors...")
    eng_vecs, eng_vocab, eng_w2i = load_glove_vectors("data/processed/glove.6B.300d.txt")
    print(f"  {len(eng_vocab)} English words, {eng_vecs.shape[1]}d")

    os.makedirs("results/figures", exist_ok=True)
    all_findings = {}

    for domain_key, domain in DOMAINS.items():
        print(f"\nAnalyzing domain: {domain['name']}...")
        findings = analyze_domain(
            domain_key, domain, sum_vecs, sum_vocab, sum_w2i,
            eng_vecs, eng_vocab, eng_w2i, figures_dir="results/figures",
        )
        all_findings[domain_key] = findings
        n_merges = len(findings.get("merges", []))
        n_splits = len(findings.get("splits", []))
        print(f"  {n_merges} merges, {n_splits} splits detected")

    plot_domain_summary(all_findings, "results/figures/domain_comparison.png")

    with open("results/geometric_findings.json", "w") as f:
        json.dump(all_findings, f, indent=2, default=str)
    print(f"\nResults written to results/geometric_findings.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_geometric_compare.py -v`
Expected: All 13 tests PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/geometric_compare.py tests/test_geometric_compare.py
git commit -m "feat: add visualization, domain runner, and full comparison pipeline"
```

---

### Task 5: Geometric Translate — Passage Extraction + Tests

**Files:**
- Create: `scripts/geometric_translate.py`
- Create: `tests/test_geometric_translate.py`

Extracts ETCSL passages for significant findings and generates geometric glosses.

- [ ] **Step 1: Write failing tests**

Create `tests/test_geometric_translate.py`:

```python
import json
import numpy as np
import pytest


def test_search_etcsl_passages():
    """Find ETCSL lines containing specific Sumerian terms."""
    from scripts.geometric_translate import search_etcsl_passages

    texts = [
        {"transliteration": "an ki-ta bad-du", "translation": "heaven from earth was separated", "line_id": "c.1.1.1.1", "source": "ETCSL"},
        {"transliteration": "lugal-e e2 mu-un-du3", "translation": "the king built the house", "line_id": "c.2.1.1.5", "source": "ETCSL"},
        {"transliteration": "nam-tar an-na", "translation": "the fate of heaven", "line_id": "c.1.1.4.3", "source": "ETCSL"},
    ]

    results = search_etcsl_passages(texts, ["an", "ki"], max_results=5)
    assert len(results) >= 1
    assert results[0]["line_id"] == "c.1.1.1.1"  # contains both "an" and "ki"


def test_geometric_gloss():
    """Generate geometric gloss for a transliteration line."""
    from scripts.geometric_translate import geometric_gloss

    sum_vecs = np.random.randn(5, 300).astype(np.float32)
    sum_vocab = ["an", "ki", "bad", "du", "e2"]
    sum_w2i = {w: i for i, w in enumerate(sum_vocab)}

    eng_vecs = np.random.randn(10, 300).astype(np.float32)
    eng_vocab = ["heaven", "earth", "separate", "build", "house",
                 "sky", "ground", "divide", "make", "temple"]
    eng_w2i = {w: i for i, w in enumerate(eng_vocab)}

    transliteration = "an ki-ta bad-du"
    gloss, evidence = geometric_gloss(
        transliteration, sum_vecs, sum_vocab, sum_w2i, eng_vecs, eng_vocab, eng_w2i, k=3,
    )
    assert isinstance(gloss, str)
    assert len(gloss) > 0
    assert isinstance(evidence, dict)
    for word, info in evidence.items():
        assert "neighbors" in info


def test_build_parallel_passage():
    """Assemble a complete parallel passage object."""
    from scripts.geometric_translate import build_parallel_passage

    passage = build_parallel_passage(
        finding_id="creation_merge_water_origin",
        domain="creation",
        source="ETCSL 1.1.1",
        line_ref="c.1.1.1.1",
        transliteration="an ki-ta bad-du",
        standard_translation="heaven from earth was separated",
        geometric_translation="sky-vault / ground-below / from / severed-apart",
        word_evidence={"an": {"neighbors": [["heaven", 0.8], ["sky", 0.7]]}},
    )
    assert passage["finding_id"] == "creation_merge_water_origin"
    assert passage["transliteration"] == "an ki-ta bad-du"
    assert "geometric_translation" in passage
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_geometric_translate.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement geometric_translate.py**

Create `scripts/geometric_translate.py`:

```python
"""
Geometric Translation: extract ETCSL passages and generate geometric glosses.

For each significant finding from geometric_compare.py, searches ETCSL texts
for relevant passages and produces three-way parallel presentations:
transliteration | standard translation | geometric translation.
"""
import json
import re

import numpy as np

from scripts.geometric_compare import DOMAINS, load_sumerian_vectors, load_glove_vectors


def search_etcsl_passages(texts, sumerian_terms, max_results=3, composition_filter=None):
    """Find ETCSL lines containing the given Sumerian terms.

    Prefers lines with multiple terms co-occurring. Optionally filters
    by composition ID prefix.
    """
    scored = []
    for entry in texts:
        translit = entry["transliteration"]
        tokens = re.split(r"[\s\-]+", translit.lower())
        matches = sum(1 for t in sumerian_terms if t.lower() in tokens)
        if matches == 0:
            continue
        if composition_filter:
            line_id = entry.get("line_id", "")
            if not any(f"c{cf}." in line_id or f"c{cf.replace('.', '')}" in line_id
                       for cf in composition_filter):
                matches *= 0.5
        scored.append((matches, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [entry for _, entry in scored[:max_results]]


def geometric_gloss(transliteration, sum_vecs, sum_vocab, sum_w2i,
                    eng_vecs, eng_vocab, eng_w2i, k=5):
    """Generate a geometric gloss: translate each Sumerian token via embedding neighbors."""
    eng_norms = np.linalg.norm(eng_vecs, axis=1, keepdims=True)
    eng_norms[eng_norms == 0] = 1
    eng_normed = eng_vecs / eng_norms
    sum_norms = np.linalg.norm(sum_vecs, axis=1, keepdims=True)
    sum_norms[sum_norms == 0] = 1
    sum_normed = sum_vecs / sum_norms

    tokens = re.split(r"[\s]+", transliteration.strip())
    gloss_parts = []
    evidence = {}

    for token in tokens:
        base = re.split(r"[\-]", token)[0].lower()
        if base in sum_w2i:
            idx = sum_w2i[base]
            sims = eng_normed @ sum_normed[idx]
            top_eng = np.argsort(sims)[::-1][:k]
            neighbors = [(eng_vocab[i], float(sims[i])) for i in top_eng]
            gloss_word = "/".join(n[0] for n in neighbors[:3])
            gloss_parts.append(gloss_word)
            evidence[base] = {"neighbors": neighbors, "original_token": token}
        else:
            gloss_parts.append(f"[{token}]")

    return " \u00b7 ".join(gloss_parts), evidence


def build_parallel_passage(finding_id, domain, source, line_ref,
                           transliteration, standard_translation,
                           geometric_translation, word_evidence):
    """Assemble a structured parallel passage object."""
    return {
        "finding_id": finding_id,
        "domain": domain,
        "source": source,
        "line_ref": line_ref,
        "transliteration": transliteration,
        "standard_translation": standard_translation,
        "geometric_translation": geometric_translation,
        "word_evidence": word_evidence,
    }


def process_findings(findings_path, etcsl_path, sum_vecs, sum_vocab, sum_w2i,
                     eng_vecs, eng_vocab, eng_w2i):
    """Process all findings and generate parallel passages."""
    with open(findings_path) as f:
        all_findings = json.load(f)
    with open(etcsl_path) as f:
        etcsl_texts = json.load(f)

    passages = []
    for domain_key, findings in all_findings.items():
        if "error" in findings:
            continue

        for merge in findings.get("merges", [])[:3]:
            pair = merge["pair"]
            finding_id = f"{domain_key}_merge_{'_'.join(pair)}"
            proxy_map = findings.get("seed_to_sumerian_proxy", {})
            sum_terms = [proxy_map.get(p, p) for p in pair]
            etcsl_compositions = DOMAINS.get(domain_key, {}).get("etcsl_compositions")
            matched = search_etcsl_passages(
                etcsl_texts, sum_terms, composition_filter=etcsl_compositions,
            )
            for entry in matched:
                gloss, evidence = geometric_gloss(
                    entry["transliteration"],
                    sum_vecs, sum_vocab, sum_w2i,
                    eng_vecs, eng_vocab, eng_w2i,
                )
                passages.append(build_parallel_passage(
                    finding_id=finding_id,
                    domain=domain_key,
                    source="ETCSL",
                    line_ref=entry["line_id"],
                    transliteration=entry["transliteration"],
                    standard_translation=entry["translation"],
                    geometric_translation=gloss,
                    word_evidence=evidence,
                ))

        for split in findings.get("splits", [])[:2]:
            pair = split["pair"]
            finding_id = f"{domain_key}_split_{'_'.join(pair)}"
            proxy_map = findings.get("seed_to_sumerian_proxy", {})
            sum_terms = [proxy_map.get(p, p) for p in pair]
            matched = search_etcsl_passages(
                etcsl_texts, sum_terms,
                composition_filter=DOMAINS.get(domain_key, {}).get("etcsl_compositions"),
            )
            for entry in matched:
                gloss, evidence = geometric_gloss(
                    entry["transliteration"],
                    sum_vecs, sum_vocab, sum_w2i,
                    eng_vecs, eng_vocab, eng_w2i,
                )
                passages.append(build_parallel_passage(
                    finding_id=finding_id,
                    domain=domain_key,
                    source="ETCSL",
                    line_ref=entry["line_id"],
                    transliteration=entry["transliteration"],
                    standard_translation=entry["translation"],
                    geometric_translation=gloss,
                    word_evidence=evidence,
                ))

    return passages


def main():
    """Run geometric translation pipeline."""
    print("Loading vectors...")
    sum_vecs, sum_vocab, sum_w2i = load_sumerian_vectors(
        "final_output/sumerian_aligned_vectors.npz",
        "final_output/sumerian_aligned_vocab.pkl",
    )
    eng_vecs, eng_vocab, eng_w2i = load_glove_vectors("data/processed/glove.6B.300d.txt")

    print("Processing findings...")
    passages = process_findings(
        "results/geometric_findings.json",
        "data/raw/etcsl_texts.json",
        sum_vecs, sum_vocab, sum_w2i,
        eng_vecs, eng_vocab, eng_w2i,
    )

    with open("results/parallel_passages.json", "w") as f:
        json.dump(passages, f, indent=2, default=str)
    print(f"Generated {len(passages)} parallel passages -> results/parallel_passages.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_geometric_translate.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/geometric_translate.py tests/test_geometric_translate.py
git commit -m "feat: add geometric translation with ETCSL passage extraction"
```

---

### Task 6: LaTeX Narrative Generator + Tests

**Files:**
- Create: `scripts/generate_narrative.py`
- Create: `tests/test_generate_narrative.py`

Assembles geometric findings, parallel passages, and figures into a compilable LaTeX essay.

- [ ] **Step 1: Write failing tests**

Create `tests/test_generate_narrative.py`:

```python
import json
import os
import pytest


def test_rank_domains():
    """Rank domains by geometric interestingness."""
    from scripts.generate_narrative import rank_domains

    findings = {
        "creation": {
            "domain": "Creation & Origin",
            "merges": [{"pair": ("a", "b"), "ratio": 3.0}],
            "splits": [],
            "centroid_displacement_neighbors": [("water", 0.5)],
            "neighborhood_divergence": {"a": 0.1, "b": 0.3},
        },
        "fate": {
            "domain": "Fate, Meaning & Purpose",
            "merges": [{"pair": ("c", "d"), "ratio": 2.5}, {"pair": ("e", "f"), "ratio": 2.1}],
            "splits": [{"pair": ("g", "h"), "ratio": 3.0}],
            "centroid_displacement_neighbors": [("name", 0.6)],
            "neighborhood_divergence": {"c": 0.05, "d": 0.1},
        },
    }
    ranked = rank_domains(findings)
    assert len(ranked) == 2
    assert ranked[0][0] == "fate"


def test_render_parallel_passage():
    """Render a parallel passage as LaTeX."""
    from scripts.generate_narrative import render_parallel_passage

    passage = {
        "transliteration": "an ki-ta bad-du",
        "standard_translation": "heaven from earth was separated",
        "geometric_translation": "sky/vault/above . ground/below/soil . from . severed/apart/divided",
        "word_evidence": {
            "an": {"neighbors": [["sky", 0.8], ["vault", 0.7], ["above", 0.6]]},
        },
    }
    latex = render_parallel_passage(passage)
    assert "\\begin{parallelpassage}" in latex
    assert "an ki-ta bad-du" in latex
    assert "heaven from earth was separated" in latex
    assert "sky/vault/above" in latex


def test_generate_latex_document(tmp_path):
    """Generate a complete compilable LaTeX document."""
    from scripts.generate_narrative import generate_latex_document

    findings = {
        "creation": {
            "domain": "Creation & Origin",
            "valid_seeds": ["create", "water", "earth"],
            "merges": [{"pair": ("create", "water"), "ratio": 2.5,
                        "eng_distance": 0.8, "sum_distance": 0.3}],
            "splits": [],
            "centroid_displacement_neighbors": [("sea", 0.5), ("deep", 0.4)],
            "neighborhood_divergence": {"create": 0.1, "water": 0.2, "earth": 0.3},
            "cluster_shape_english": {"eigenvalues": [0.5, 0.3, 0.1], "effective_dim": 2.1},
            "cluster_shape_sumerian": {"eigenvalues": [0.7, 0.2, 0.05], "effective_dim": 1.6},
            "seed_to_sumerian_proxy": {"create": "dim2", "water": "a", "earth": "ki"},
            "cluster": {"sumerian_words": {"dim2": {"seed": "create", "similarity": 0.6}},
                        "discovered_english": ["sea"]},
        },
    }
    passages = [{
        "finding_id": "creation_merge_create_water",
        "domain": "creation",
        "source": "ETCSL",
        "line_ref": "c.1.1.1.1",
        "transliteration": "a-ab-ba ki gal-la",
        "standard_translation": "the sea in the great place",
        "geometric_translation": "water/sea/deep . ground/earth/place . great/vast/wide",
        "word_evidence": {"a": {"neighbors": [["water", 0.8]]}},
    }]

    out_dir = str(tmp_path)
    tex_path = generate_latex_document(findings, passages, str(tmp_path / "figures"), out_dir)
    assert os.path.exists(tex_path)
    content = open(tex_path).read()
    assert "\\documentclass" in content
    assert "\\begin{document}" in content
    assert "\\end{document}" in content
    assert "parallelpassage" in content
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_generate_narrative.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement generate_narrative.py**

Create `scripts/generate_narrative.py`:

```python
"""
Narrative Generator: assemble geometric findings into a LaTeX essay.

Produces a long-form philosophical essay with embedded UMAP visualizations,
distance heatmap diffs, and three-column parallel passage presentations.
"""
import json
import os
import subprocess


LATEX_PREAMBLE = r"""\documentclass[11pt, oneside]{article}
\usepackage[margin=1.2in]{geometry}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{microtype}
\usepackage{paracol}
\usepackage{booktabs}
\usepackage[T1]{fontenc}
\usepackage{tgpagella}

\definecolor{translit}{HTML}{2A5A8A}
\definecolor{standard}{HTML}{333333}
\definecolor{geometric}{HTML}{8A2A2A}

\newenvironment{parallelpassage}{%
  \begin{paracol}{3}%
  \setlength{\columnsep}{1.5em}%
}{%
  \end{paracol}%
  \vspace{1em}%
}

\setlength{\parindent}{0pt}
\setlength{\parskip}{0.8em}

\pagestyle{plain}

\begin{document}
"""

LATEX_POSTAMBLE = r"""
\end{document}
"""


def rank_domains(findings):
    """Rank domains by geometric interestingness."""
    scored = []
    for key, f in findings.items():
        if "error" in f:
            continue
        n_merges = len(f.get("merges", []))
        n_splits = len(f.get("splits", []))
        avg_div = 0
        divs = f.get("neighborhood_divergence", {})
        if divs:
            avg_div = 1 - (sum(divs.values()) / len(divs))
        score = n_merges * 2 + n_splits * 2 + avg_div * 5
        scored.append((key, f, score))
    scored.sort(key=lambda x: x[2], reverse=True)
    return [(key, f) for key, f, _ in scored]


def escape_latex(text):
    """Escape special LaTeX characters."""
    for char in ["&", "%", "$", "#", "_"]:
        text = text.replace(char, f"\\{char}")
    return text


def render_parallel_passage(passage):
    """Render a single parallel passage as LaTeX."""
    translit = escape_latex(passage["transliteration"])
    standard = escape_latex(passage["standard_translation"])
    geometric = escape_latex(passage["geometric_translation"])

    return rf"""
\begin{{parallelpassage}}
\textcolor{{translit}}{{\textit{{{translit}}}}}
\switchcolumn
\textcolor{{standard}}{{{standard}}}
\switchcolumn
\textcolor{{geometric}}{{\textbf{{{geometric}}}}}
\end{{parallelpassage}}
"""


def render_word_evidence(passage):
    """Render word evidence as a compact LaTeX table."""
    evidence = passage.get("word_evidence", {})
    if not evidence:
        return ""

    rows = []
    for word, info in evidence.items():
        neighbors = info.get("neighbors", [])[:5]
        neighbor_str = ", ".join(f"{n[0]} ({n[1]:.2f})" for n in neighbors)
        word_escaped = escape_latex(word)
        rows.append(rf"  \texttt{{{word_escaped}}} & {neighbor_str} \\")

    if not rows:
        return ""

    return rf"""
\begin{{small}}
\begin{{tabular}}{{ll}}
\toprule
\textbf{{Sumerian}} & \textbf{{Geometric Neighbors (cosine similarity)}} \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{small}}
"""


def render_domain_section(domain_key, findings, passages, figures_dir):
    """Render a full domain section of the essay."""
    domain_name = findings["domain"]
    merges = findings.get("merges", [])
    splits = findings.get("splits", [])
    disp_neighbors = findings.get("centroid_displacement_neighbors", [])
    seed_proxy = findings.get("seed_to_sumerian_proxy", {})
    eng_shape = findings.get("cluster_shape_english", {})
    sum_shape = findings.get("cluster_shape_sumerian", {})
    domain_passages = [p for p in passages if p["domain"] == domain_key]

    sections = []

    sections.append(rf"""
\vspace{{2em}}
\begin{{center}}\rule{{0.5\textwidth}}{{0.4pt}}\end{{center}}
\vspace{{1em}}

{{\Large \textbf{{{domain_name}}}}}
\vspace{{0.5em}}
""")

    umap_path = os.path.join(figures_dir, f"{domain_key}_umap.png")
    if os.path.exists(umap_path):
        sections.append(rf"""
\begin{{center}}
\includegraphics[width=0.9\textwidth]{{{umap_path}}}
\end{{center}}
""")

    if merges:
        sections.append(r"\textbf{What Sumerian merges together:}" + "\n\n")
        for m in merges[:3]:
            a, b = m["pair"]
            ratio = m["ratio"]
            sum_a = escape_latex(seed_proxy.get(a, "?"))
            sum_b = escape_latex(seed_proxy.get(b, "?"))
            sections.append(
                rf"In English, \textit{{{a}}} and \textit{{{b}}} sit far apart "
                rf"(cosine distance {m['eng_distance']:.2f}). In Sumerian space, "
                rf"their proxies \texttt{{{sum_a}}} and \texttt{{{sum_b}}} collapse together "
                rf"(distance {m['sum_distance']:.2f}) --- a {ratio:.1f}$\times$ compression. "
                rf"Sumerian treats these as aspects of the same concept."
                "\n\n"
            )

    if splits:
        sections.append(r"\textbf{What Sumerian pulls apart:}" + "\n\n")
        for s in splits[:3]:
            a, b = s["pair"]
            ratio = s["ratio"]
            sections.append(
                rf"English places \textit{{{a}}} and \textit{{{b}}} nearby "
                rf"(distance {s['eng_distance']:.2f}), but Sumerian separates them "
                rf"(distance {s['sum_distance']:.2f}) --- a {ratio:.1f}$\times$ expansion. "
                rf"A distinction English has lost."
                "\n\n"
            )

    if disp_neighbors:
        direction_words = ", ".join(f"\\textit{{{w}}}" for w, _ in disp_neighbors[:5])
        sections.append(
            rf"The Sumerian understanding of this domain is displaced toward: "
            rf"{direction_words}. This is where the center of gravity shifts "
            rf"when you think about {domain_name.lower()} in Sumerian rather than English."
            "\n\n"
        )

    if eng_shape and sum_shape:
        eng_dim = eng_shape.get("effective_dim", 0)
        sum_dim = sum_shape.get("effective_dim", 0)
        if sum_dim < eng_dim * 0.8:
            sections.append(
                rf"The Sumerian cluster is more compressed (effective dimensionality "
                rf"{sum_dim:.1f} vs English {eng_dim:.1f}), suggesting these concepts "
                rf"are organized along fewer axes --- a more unified understanding."
                "\n\n"
            )
        elif sum_dim > eng_dim * 1.2:
            sections.append(
                rf"The Sumerian cluster is more expansive (effective dimensionality "
                rf"{sum_dim:.1f} vs English {eng_dim:.1f}), suggesting finer internal "
                rf"distinctions than English makes."
                "\n\n"
            )

    heatmap_path = os.path.join(figures_dir, f"{domain_key}_distance_diff.png")
    if os.path.exists(heatmap_path):
        sections.append(rf"""
\begin{{center}}
\includegraphics[width=0.85\textwidth]{{{heatmap_path}}}
\end{{center}}
""")

    if domain_passages:
        sections.append(r"""
\vspace{1em}
\textbf{Parallel Passages:} \textcolor{translit}{transliteration} | \textcolor{standard}{standard translation} | \textcolor{geometric}{geometric translation}
\vspace{0.5em}
""")
        for passage in domain_passages[:3]:
            sections.append(render_parallel_passage(passage))
            sections.append(render_word_evidence(passage))

    return "\n".join(sections)


def render_opening():
    """Opening section of the essay."""
    return r"""
\begin{center}
{\LARGE \textbf{The Shape of Meaning}}

\vspace{0.3em}
{\large Geometric Translation and the Topology of Ancient Thought}

\vspace{2em}
\end{center}

What if translation is not just imprecise but \textit{dimensionally impoverished}?

When we translate Sumerian into English, we map each word to its nearest dictionary entry --- \textit{nam.tar} becomes ``fate,'' \textit{me} becomes ``divine decree,'' \textit{an} becomes ``heaven.'' Each translation is a point-to-point correspondence. But meaning is not a point. Meaning is a \textit{position in a space} --- defined not by what a word denotes but by what surrounds it, what it is near to, what it is far from.

Modern computational linguistics has given us a way to see this space. When a neural network learns to predict words from context, it builds an internal geometry where proximity encodes meaning. Words that appear in similar contexts cluster together. The relationships between words become vectors --- directions in a high-dimensional space. The famous demonstration: \textit{king} minus \textit{man} plus \textit{woman} equals \textit{queen}. Not because the machine understands monarchy, but because the geometry of usage encodes the relationship.

We have done something unusual with this machinery. We trained it on the largest available corpus of Sumerian transliterated text --- literary compositions, administrative records, royal hymns, mythological narratives spanning two millennia of Mesopotamian civilization. Then we aligned the resulting geometric space with modern English, using dictionary entries and parallel translations as anchor points.

The alignment lets us ask a new kind of question. Not ``what does this word mean?'' but ``\textit{what shape does this concept occupy},'' and how does that shape differ between a mind that thinks in Sumerian and a mind that thinks in English?

What follows is a report on what the geometry reveals. For three conceptual domains --- creation, fate, and selfhood --- we compare the topology of Sumerian semantic space against English. Where the shapes diverge, something has changed in how humans construct meaning. Where they converge, something may be universal.

The method is simple. The implications are not.
"""


def render_synthesis(findings):
    """Synthesis section tying the domains together."""
    total_merges = sum(len(f.get("merges", [])) for f in findings.values() if "error" not in f)
    total_splits = sum(len(f.get("splits", [])) for f in findings.values() if "error" not in f)

    return rf"""
\vspace{{2em}}
\begin{{center}}\rule{{0.5\textwidth}}{{0.4pt}}\end{{center}}
\vspace{{1em}}

{{\Large \textbf{{Synthesis}}}}
\vspace{{0.5em}}

Across the three domains, we find {total_merges} geometric merges (concepts that Sumerian treats as unified but English separates) and {total_splits} splits (distinctions Sumerian makes that English has collapsed).

These are not translation errors. They are structural differences in how meaning is organized. The merges tell us what we have pulled apart --- concepts that once inhabited the same region of semantic space, understood as aspects of a single phenomenon, now scattered across our vocabulary as if they were always separate. The splits tell us the opposite: fine distinctions that Sumerian speakers maintained, collapsed in modern English into a single blurred concept.

The embedding manifold has not simply shrunk or expanded. It has \textit{{rotated}}. The axes along which meaning is organized have shifted. What was a single dimension of experience in Sumerian --- say, the axis connecting naming, fate, and cosmic order --- has been decomposed in English into separate, seemingly unrelated concepts.

If the manifold has rotated, there is an axis of rotation. That axis is the invariant --- what stayed fixed while everything else transformed. And the gap between the two orientations defines a direction: the trajectory of human consciousness across four thousand years, expressed not as philosophy but as geometry.

We do not claim this trajectory is good or bad. We claim it is \textit{{measurable}}. And what can be measured can be studied, compared, and understood --- not as nostalgia for a lost golden age, but as cartography of how the human capacity for meaning has evolved.

What have we gained? The dimensionality of selfhood. The precision of mechanical causation. The vast conceptual territory of science.

What have we lost? That is harder to say, because the loss is precisely in the territory we can no longer easily think about. But the geometry shows us where to look: in the collapsed regions, the flattened curvatures, the voids where Sumerian once had rich structure and English has only silence.
"""


def render_closing():
    """Closing section of the essay."""
    return r"""
\vspace{2em}
\begin{center}\rule{0.5\textwidth}{0.4pt}\end{center}
\vspace{1em}

{\Large \textbf{Coda}}
\vspace{0.5em}

There is a direction implied by the rotation. A vector from Sumerian to English, from ancient to modern, from one shape of meaning to another. If we added intermediate points --- Akkadian, Classical Greek, Latin, Old English --- we could trace the curve of that rotation over time.

Is it smooth? Is it accelerating? Did it lurch at certain moments --- the invention of writing, the emergence of monotheism, the printing press, the internet?

That curve would describe the trajectory of human consciousness as a mathematical object. Not derived from ideology or philosophy, but from the shapes that meaning takes when embedded in language.

We have computed two points on that curve. The distance between them is four thousand years.

The rest is ahead.
"""


def generate_latex_document(findings, passages, figures_dir, output_dir):
    """Generate complete LaTeX document."""
    os.makedirs(output_dir, exist_ok=True)

    ranked = rank_domains(findings)

    body_sections = []
    body_sections.append(render_opening())

    for domain_key, domain_findings in ranked:
        body_sections.append(
            render_domain_section(domain_key, domain_findings, passages, figures_dir)
        )

    body_sections.append(render_synthesis(findings))
    body_sections.append(render_closing())

    full_doc = LATEX_PREAMBLE + "\n".join(body_sections) + LATEX_POSTAMBLE

    tex_path = os.path.join(output_dir, "geometric_narrative.tex")
    with open(tex_path, "w") as f:
        f.write(full_doc)

    # Try to compile PDF
    try:
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode",
             "-output-directory", output_dir, tex_path],
            capture_output=True, timeout=60,
        )
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode",
             "-output-directory", output_dir, tex_path],
            capture_output=True, timeout=60,
        )
        pdf_path = os.path.join(output_dir, "geometric_narrative.pdf")
        if os.path.exists(pdf_path):
            print(f"PDF compiled -> {pdf_path}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("pdflatex not available or timed out -- .tex file generated, compile manually")

    return tex_path


def main():
    """Assemble narrative from findings and passages."""
    print("Loading findings...")
    with open("results/geometric_findings.json") as f:
        findings = json.load(f)
    with open("results/parallel_passages.json") as f:
        passages = json.load(f)

    print("Generating LaTeX document...")
    tex_path = generate_latex_document(findings, passages, "results/figures", "output")
    print(f"LaTeX written -> {tex_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_generate_narrative.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_narrative.py tests/test_generate_narrative.py
git commit -m "feat: add LaTeX narrative generator with parallel passages"
```

---

### Task 7: End-to-End Integration Test

**Files:**
- Create: `tests/test_narrative_integration.py`

Runs the full pipeline with synthetic data to verify all three scripts chain together.

- [ ] **Step 1: Write integration test**

Create `tests/test_narrative_integration.py`:

```python
"""End-to-end integration test for the geometric narrative pipeline."""
import json
import os
import pickle

import numpy as np
import pytest


@pytest.fixture
def synthetic_env(tmp_path):
    """Create a self-contained synthetic environment for the full pipeline."""
    n_sum = 50
    sum_vocab = (
        ["lugal", "e2", "dingir", "nam", "an", "ki", "a", "bad",
         "du", "me", "tar", "zi", "sa", "dim2", "kur", "abzu"]
        + [f"sum{i}" for i in range(n_sum - 16)]
    )
    sum_vecs = np.random.randn(n_sum, 300).astype(np.float32)
    sum_vecs[4] = sum_vecs[5] + np.random.randn(300) * 0.1  # an near ki

    vec_path = str(tmp_path / "sum_vectors.npz")
    np.savez(vec_path, vectors=sum_vecs)

    vocab_path = str(tmp_path / "sum_vocab.pkl")
    with open(vocab_path, "wb") as f:
        pickle.dump(sum_vocab, f)

    eng_seeds = [
        "create", "begin", "birth", "origin", "emerge", "form",
        "earth", "water", "heaven", "separate", "divide", "first",
        "fate", "destiny", "purpose", "decree", "meaning", "life",
        "death", "name", "order", "tablet", "judge", "decide",
        "self", "soul", "spirit", "mind", "heart", "body",
        "breath", "shadow", "dream", "blood", "eye", "inner",
        "law", "divine", "chaos", "primordial", "thought", "will",
    ]
    n_eng = 100
    eng_vocab = eng_seeds + [f"eng{i}" for i in range(n_eng - len(eng_seeds))]
    eng_vecs = np.random.randn(n_eng, 300).astype(np.float32)

    glove_path = str(tmp_path / "glove_test.txt")
    with open(glove_path, "w") as f:
        for word, vec in zip(eng_vocab, eng_vecs):
            f.write(word + " " + " ".join(f"{v:.6f}" for v in vec) + "\n")

    etcsl = [
        {"transliteration": "an ki-ta bad-du", "translation": "heaven from earth was separated",
         "line_id": "c.1.1.1.1", "source": "ETCSL"},
        {"transliteration": "nam tar-re me an-na", "translation": "fate decrees the divine powers of heaven",
         "line_id": "c.1.1.4.3", "source": "ETCSL"},
        {"transliteration": "zi sa-ga dingir-re", "translation": "the breath and heart of the gods",
         "line_id": "c.1.8.1.4.5", "source": "ETCSL"},
        {"transliteration": "a abzu-ta e3-a", "translation": "water rising from the abyss",
         "line_id": "c.1.1.1.10", "source": "ETCSL"},
        {"transliteration": "lugal-e nam mu-un-tar", "translation": "the king decreed the fate",
         "line_id": "c.1.3.1.20", "source": "ETCSL"},
    ]
    etcsl_path = str(tmp_path / "etcsl_texts.json")
    with open(etcsl_path, "w") as f:
        json.dump(etcsl, f)

    return {
        "vec_path": vec_path, "vocab_path": vocab_path,
        "glove_path": glove_path, "etcsl_path": etcsl_path,
        "tmp_path": tmp_path,
    }


def test_full_pipeline(synthetic_env):
    """Run geometric_compare -> geometric_translate -> generate_narrative end-to-end."""
    from scripts.geometric_compare import (
        load_sumerian_vectors, load_glove_vectors, analyze_domain, DOMAINS,
    )
    from scripts.geometric_translate import (
        search_etcsl_passages, geometric_gloss, build_parallel_passage,
    )
    from scripts.generate_narrative import generate_latex_document

    env = synthetic_env
    tmp = env["tmp_path"]

    # Stage 1: Geometric comparison
    sum_vecs, sum_vocab, sum_w2i = load_sumerian_vectors(env["vec_path"], env["vocab_path"])
    eng_vecs, eng_vocab, eng_w2i = load_glove_vectors(env["glove_path"])

    figures_dir = str(tmp / "figures")
    os.makedirs(figures_dir, exist_ok=True)

    all_findings = {}
    for domain_key, domain in DOMAINS.items():
        findings = analyze_domain(
            domain_key, domain, sum_vecs, sum_vocab, sum_w2i,
            eng_vecs, eng_vocab, eng_w2i, figures_dir=figures_dir,
        )
        all_findings[domain_key] = findings

    findings_path = str(tmp / "findings.json")
    with open(findings_path, "w") as f:
        json.dump(all_findings, f, default=str)

    assert len(all_findings) == 3
    for key in ["creation", "fate", "self"]:
        assert key in all_findings

    # Stage 2: Geometric translation
    with open(env["etcsl_path"]) as f:
        etcsl_texts = json.load(f)

    passages = []
    for domain_key, findings in all_findings.items():
        if "error" in findings:
            continue
        for merge in findings.get("merges", [])[:2]:
            pair = merge["pair"]
            proxy_map = findings.get("seed_to_sumerian_proxy", {})
            terms = [proxy_map.get(p, p) for p in pair]
            matched = search_etcsl_passages(etcsl_texts, terms)
            for entry in matched:
                gloss, evidence = geometric_gloss(
                    entry["transliteration"],
                    sum_vecs, sum_vocab, sum_w2i,
                    eng_vecs, eng_vocab, eng_w2i,
                )
                passages.append(build_parallel_passage(
                    finding_id=f"{domain_key}_merge_{'_'.join(pair)}",
                    domain=domain_key, source="ETCSL",
                    line_ref=entry["line_id"],
                    transliteration=entry["transliteration"],
                    standard_translation=entry["translation"],
                    geometric_translation=gloss, word_evidence=evidence,
                ))

    # Stage 3: Generate narrative
    output_dir = str(tmp / "output")
    tex_path = generate_latex_document(all_findings, passages, figures_dir, output_dir)

    assert os.path.exists(tex_path)
    content = open(tex_path).read()
    assert "\\documentclass" in content
    assert "\\begin{document}" in content
    assert "\\end{document}" in content
    assert "Shape of Meaning" in content
    assert "Creation" in content or "Fate" in content or "Self" in content
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/test_narrative_integration.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_narrative_integration.py
git commit -m "test: add end-to-end integration test for geometric narrative pipeline"
```

---

### Task 8: Run Full Pipeline on Real Data

**Files:**
- No new files

- [ ] **Step 1: Run geometric comparison**

Run: `python scripts/geometric_compare.py`
Expected: creates `results/geometric_findings.json` and 7 PNGs in `results/figures/`. Prints merge/split counts per domain.

- [ ] **Step 2: Inspect findings**

Run: `python -c "import json; f=json.load(open('results/geometric_findings.json')); [print(k, len(f[k].get('merges',[])), 'merges', len(f[k].get('splits',[])), 'splits') for k in f]"`
Expected: counts for each domain. If any domain has 0 merges and 0 splits, lower the threshold in `detect_merges_splits` from 2.0 to 1.5 and rerun.

- [ ] **Step 3: Run geometric translation**

Run: `python scripts/geometric_translate.py`
Expected: creates `results/parallel_passages.json`.

- [ ] **Step 4: Run narrative generator**

Run: `python scripts/generate_narrative.py`
Expected: creates `output/geometric_narrative.tex` and attempts PDF compilation.

- [ ] **Step 5: Verify outputs**

Run: `ls -la output/ && ls -la results/figures/ && python -c "import json; p=json.load(open('results/parallel_passages.json')); print(len(p), 'passages')"`

- [ ] **Step 6: Add generated dirs to gitignore and commit**

```bash
echo -e "\n# Generated results\nresults/\noutput/" >> .gitignore
git add .gitignore
git commit -m "chore: add results and output dirs to gitignore"
```

---

### Task 9: Final Review + Polish

**Files:**
- Possibly modify: any of the three scripts if issues found during real-data run

- [ ] **Step 1: Compile and inspect the PDF**

If PDF was not auto-compiled:
Run: `pdflatex -output-directory output output/geometric_narrative.tex && pdflatex -output-directory output output/geometric_narrative.tex`

Open and visually inspect.

- [ ] **Step 2: Fix any LaTeX compilation errors**

Common issues: unescaped special characters in Sumerian words, missing figure paths, overfull hboxes in parallel columns. Fix in `generate_narrative.py` and rerun.

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests pass.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete geometric narrative pipeline (compare -> translate -> LaTeX)"
```
