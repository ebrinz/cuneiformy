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

**v1 Baseline** &mdash; direct port of [heiroglyphy](https://github.com/ebrinz/heiroglyphy) V15 architecture to Sumerian:

| Metric | Cuneiformy (Sumerian) | Heiroglyphy (Egyptian) |
|--------|:--------------------:|:---------------------:|
| Top-1 Accuracy | **17.30%** | 32.35% |
| Top-5 Accuracy | **22.90%** | 41.47% |
| Top-10 Accuracy | **25.19%** | 45.13% |
| Training Anchors | 1,572 | 5,360 |
| Corpus Lines | 2.8M (pre-dedup) | 100K |
| Optimal Alpha | 100 | 0.001 |

The accuracy gap is driven by **anchor coverage** (1,572 vs 5,360 training pairs), not corpus size. Sumerian's agglutinative morphology creates a mismatch between ORACC dictionary citation forms and the hyphen-split ATF corpus tokens.

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
    vectors_path="final_output/sumerian_aligned_vectors.npz",
    vocab_path="final_output/sumerian_aligned_vocab.pkl",
    glove_vectors=glove_vectors,
    glove_vocab=glove_vocab,
)

# Find Sumerian words for an English concept
lookup.find("king")      # -> [("lugal", 0.72), ...]
lookup.find("water")     # -> [("a", 0.58), ...]

# Vector analogy: king is to queen as god is to ?
lookup.find_analogy("king", "queen", "god")

# Weighted blend of concepts
lookup.find_blend({"sun": 0.7, "power": 0.3})
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
pytest tests/ -v    # 34 tests
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
