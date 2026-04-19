<p align="center">
  <img src="assets/banner.svg" alt="Cuneiformy" width="100%"/>
</p>

<p align="center">
  <strong>Map ancient Sumerian words into modern English semantic space</strong>
</p>

<p align="center">
  <a href="#results">Results</a> &bull;
  <a href="#how-it-works">How It Works</a> &bull;
  <a href="#usage">Usage</a> &bull;
  <a href="#running-the-pipeline">Pipeline</a> &bull;
  <a href="#data-sources">Data Sources</a>
</p>

---

## Results

Current best — whitened EmbeddingGemma alignment (post-Phase-B):

| Metric | Cuneiformy (whitened-Gemma 768d) | Cuneiformy v1 (GloVe 300d) | Heiroglyphy (Egyptian) |
|--------|:--------------------:|:---------------------:|:---------------------:|
| Top-1 Accuracy | **19.85%** | 17.30% | 32.35% |
| Top-5 Accuracy | **23.66%** | 22.90% | 41.47% |
| Top-10 Accuracy | **26.21%** | 25.19% | 45.13% |
| Training Anchors | 1,572 | 1,572 | 5,360 |
| Corpus Lines | 2.8M (pre-dedup) | 2.8M | 100K |
| Target Space | 768d whitened-Gemma (primary) | 300d GloVe (secondary, retained) | 300d GloVe |

Both alignment targets are accessible via one `SumerianLookup` class (`space="gemma"|"glove"`).

The remaining gap vs Heiroglyphy is driven by **anchor coverage** (1,572 vs 5,360 training pairs), not corpus size. Ongoing work is diagnosed and sequenced in the [experiment journal](docs/EXPERIMENT_JOURNAL.md).

### Research progress

Active experiment log: [`docs/EXPERIMENT_JOURNAL.md`](docs/EXPERIMENT_JOURNAL.md). Recent findings (newest first):

- **2026-04-19 — Workstream 2b-pre:** Coverage diagnostic attributes **64.85%** of the 11,798 `sumerian_vocab_miss` anchors to a simple ASCII-normalization gap between the anchor extractor and the corpus tokenizer (subscripts → ASCII, strip determinative braces, drop hyphens). Expected next-step: a ~20-line fix to `scripts/06_extract_anchors.py` lifts training-anchor count ~5× without any retrain. Inference-based alternatives (FastText subword inference, morpheme composition) recover far fewer anchors with semantically-correct projections (10.7% and 1.8% Tier-2 top-5 accuracy) — not the next lever to pull.
- **2026-04-18 — Workstream 2a:** Anchor audit baselined valid-anchor survival at 14.05% (1,951/13,886). 84.96% of all dropout is `sumerian_vocab_miss`; every other bucket combined is under 1%.
- **2026-04-16 — Phase B:** Dual-view Sumerian lookup. Whitened EmbeddingGemma and GloVe now coexist as parallel alignment targets; downstream code toggles via `space="gemma"|"glove"`.
- **2026-04-16 — Phase A retry #2:** BERT-whitening (Su et al. 2021) applied to the EmbeddingGemma target unlocked +2.54pp top-1 over GloVe. Centering + whitening is mandatory for any contextual-encoder alignment target.

## How It Works

```
Sumerian corpus (ETCSL + CDLI + ORACC)
        |
  ATF cleaning & tokenization
        |
  FastText skip-gram (768d)
        |
  Zero-padding fusion (768d + 768d = 1536d)
        |
  Ridge Regression (alpha=100)
        |
  GloVe 300d English space
        |
  Nearest-neighbor retrieval
```

The approach follows a cross-lingual embedding alignment strategy:

1. **Train monolingual embeddings** on a large Sumerian corpus using FastText
2. **Fuse** text embeddings with zero-padding (dimensionality regularization)
3. **Learn a linear mapping** from Sumerian embedding space to English GloVe space using anchor word pairs (Sumerian words with known English translations)
4. **Evaluate** by checking if the nearest English neighbor of a projected Sumerian vector is the correct translation

## Usage

```python
from final_output.sumerian_lookup import SumerianLookup

lookup = SumerianLookup(
    gemma_vectors_path="final_output/sumerian_aligned_gemma_vectors.npz",
    glove_vectors_path="final_output/sumerian_aligned_vectors.npz",
    vocab_path="final_output/sumerian_aligned_vocab.pkl",
    gemma_english_path="models/english_gemma_whitened_768d.npz",
    glove_english_vectors=glove_vectors,
    glove_english_vocab=glove_vocab,
)

# Default space is the whitened-Gemma 768d manifold:
lookup.find("king")                     # -> [("ul3", 0.67), ("asal", 0.51), ...]

# Query the GloVe 300d manifold:
lookup.find("king", space="glove")      # -> [("ul3", 0.68), ("se2", 0.60), ...]

# Both spaces at once:
lookup.find_both("fate")                # -> {"gemma": [...], "glove": [...]}

# Vector analogy in either space:
lookup.find_analogy("king", "queen", "god", space="gemma")

# Weighted blend of concepts:
lookup.find_blend({"sun": 0.7, "power": 0.3}, space="gemma")
```

## Running the Pipeline

### Prerequisites

```bash
pip install -r requirements.txt
```

### Full Pipeline

```bash
# 1. Scrape corpora (ETCSL ~5MB, CDLI ~230MB, ORACC ~700MB)
python scripts/01_scrape_etcsl.py
python scripts/02_scrape_cdli.py
python scripts/03_scrape_oracc.py

# 2. Process corpus
python scripts/04_deduplicate_corpus.py
python scripts/05_clean_and_tokenize.py
python scripts/06_extract_anchors.py

# 3. Download GloVe (862MB, or symlinks from heiroglyphy)
python scripts/download_glove.py

# 4. Train and evaluate
python scripts/07_train_fasttext.py     # ~30-60 min
python scripts/08_fuse_embeddings.py
python scripts/09_align_and_evaluate.py
python scripts/10_export_production.py
```

### Tests

```bash
pytest tests/ --ignore=tests/test_integration.py -v    # 110 tests
```

## Data Sources

| Source | Content | Size |
|--------|---------|------|
| [ETCSL](https://etcsl.orinst.ox.ac.uk/) | Sumerian literary compositions with English translations | 36K lines, 35K with translations |
| [CDLI](https://cdli.ucla.edu/) | Bulk Sumerian transliterations (ATF format) | 96K texts, 1.4M lines |
| [ORACC](https://oracc.museum.upenn.edu/) | Lemmatized Sumerian with English glosses | 90K texts, 4.3M lemmas, 2.8K unique glosses |
| [GloVe 6B](https://nlp.stanford.edu/projects/glove/) | Pre-trained English word vectors | 400K words, 300d |

## Architecture

Mirrors [heiroglyphy](https://github.com/ebrinz/heiroglyphy) V15 with Sumerian-specific adaptations:

| Parameter | Value |
|-----------|-------|
| FastText dimensions | 768 |
| FastText window | 10 |
| FastText min_count | 5 |
| FastText algorithm | skip-gram |
| Fusion | 768d text + 768d zero-padding |
| Alignment | Ridge Regression |
| Optimal alpha | 100 |
| Target space | GloVe 6B 300d |

### Key Finding

The optimal Ridge alpha is **100** (vs heiroglyphy's 0.001). With only 1,572 training anchors and 1,536 dimensions, the system is underdetermined &mdash; higher regularization prevents overfitting. Zero-padding has no measurable effect at this sample size.

## Project Structure

```
cuneiformy/
├── scripts/           # Numbered pipeline scripts (01-10)
├── data/              # Raw and processed data (gitignored)
├── models/            # Trained FastText models (gitignored)
├── results/           # Evaluation results
├── final_output/      # Production vectors + lookup API
├── tests/             # 34 unit + integration tests
└── docs/              # Design spec and implementation plan
```

## License

MIT
