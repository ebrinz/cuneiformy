# Cuneiformy: Sumerian-English Embedding Alignment

## Overview

Transpose the heiroglyphy V15 SOTA model (32.35% Top-1 on Egyptian) to Sumerian cuneiform texts. Direct port of the proven architecture: FastText 768d + 768d zero-padding → Ridge Regression → GloVe 300d English space.

## Data Pipeline

### Corpus Sources (3 sources, deduplicated)

1. **ETCSL (Electronic Text Corpus of Sumerian Literature)** — ~400 literary compositions with transliterations and English translations. Primary anchor source via parallel text co-occurrence.
2. **CDLI (Cuneiform Digital Library Initiative)** — ~350K+ texts. Bulk corpus. Filter to Sumerian-only (exclude Akkadian, Hittite, etc.).
3. **ORACC (Open Richly Annotated Cuneiform Corpus)** — curated transliterations from sub-projects (DCCLT, ePSD2 corpus texts, etc.).

### Deduplication

CDLI P-numbers are the canonical identifier. ETCSL and ORACC texts frequently carry CDLI numbers — use these for cross-source deduplication.

### Anchor Extraction (two methods, merged)

1. **ePSD (Electronic Pennsylvania Sumerian Dictionary)** — direct Sumerian-English dictionary entries (~7K+ entries). High-confidence anchors.
2. **ETCSL parallel translations** — co-occurrence analysis on Sumerian transliterations paired with English translations. Supplementary anchors with confidence scoring.

Merge, deduplicate, and confidence-score the combined anchor set. Target: 7K-10K anchors (comparable to heiroglyphy's 8,541).

### Corpus Cleaning

- Replace ATF (ASCII Transliteration Format) separators with spaces
- Normalize whitespace (collapse multiple spaces)
- Strip editorial markers: `[...]` (restoration), `(...)` (interpolation), `#` (damaged), `!` (correction), `?` (uncertain)
- Output: `cleaned_corpus.txt` (one sentence per line, space-separated tokens)

## Embedding & Alignment Architecture

Direct mirror of heiroglyphy V15:

```
Sumerian word (transliterated)
        |
FastText skip-gram (768d)
  min_count=5, window=10, epochs=10
        |
Concatenate with zero-padding (768d)
  = 1536d fused vector
        |
Ridge Regression (alpha=0.001)
  trained on 80% of anchor pairs (random_state=42)
        |
GloVe 300d English space
        |
Nearest-neighbor retrieval (cosine similarity)
```

### Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| FastText vector_size | 768 | heiroglyphy V15 |
| FastText min_count | 5 | heiroglyphy V15 (filters noise, proven optimal) |
| FastText window | 10 | heiroglyphy V15 (wide context for short texts) |
| FastText epochs | 10 | heiroglyphy V15 |
| FastText sg | 1 | skip-gram (outperformed CBOW in heiroglyphy) |
| Zero-padding dim | 768 | heiroglyphy V15 (regularization via dimensionality) |
| Ridge alpha | 0.001 | heiroglyphy V15 (minimum regularization, best retrieval) |
| Train/test split | 80/20 | heiroglyphy V15 (random_state=42) |
| GloVe | 6B 300d | same target space as heiroglyphy |

### Evaluation

- Top-1, Top-5, Top-10 accuracy on held-out 20% test anchors
- Direct comparison with heiroglyphy Egyptian results
- Domain-specific semantic validation (deity names, place names, common nouns)

## Production Output

- `sumerian_aligned_vectors.npz` — all vocab words projected to 300d (float16)
- `sumerian_aligned_vocab.pkl` — word-to-index mapping
- `sumerian_lookup.py` — semantic search API: find, find_analogy, find_blend
- `metadata.json` — full configuration, results, corpus stats

## Project Structure

```
cuneiformy/
├── scripts/
│   ├── 01_scrape_etcsl.py
│   ├── 02_scrape_cdli.py
│   ├── 03_scrape_oracc.py
│   ├── 04_deduplicate_corpus.py
│   ├── 05_clean_and_tokenize.py
│   ├── 06_extract_anchors.py
│   ├── 07_train_fasttext.py
│   ├── 08_fuse_embeddings.py
│   ├── 09_align_and_evaluate.py
│   └── 10_export_production.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── dictionaries/
├── models/
├── results/
├── final_output/
│   ├── sumerian_aligned_vectors.npz
│   ├── sumerian_aligned_vocab.pkl
│   ├── sumerian_lookup.py
│   └── metadata.json
└── requirements.txt
```

## Dependencies

numpy, scipy, scikit-learn, gensim, pandas, requests, beautifulsoup4

## Future Work (not in scope for this spec)

- Morphology-aware embeddings (shorter subword n-grams for Sumerian agglutination)
- Hyperparameter sweep tuned for Sumerian corpus characteristics
- Real cuneiform sign image features (replacing zero-padding with CNN features)
