# Hyper-glyphy Monorepo Reorganization — Design Spec

**Date:** 2026-05-03
**Status:** approved
**Approach:** Big-bang `git mv` (Approach A — two commits: structural moves, then import fixes)

---

## Goal

Restructure `cuneiformy` into `hyper-glyphy`: a monorepo that hosts multiple ancient-language embedding spaces as siblings under one framework. Phase α moves Sumerian, extracts the civilization-agnostic framework, and separates shared English-target artifacts. The GitHub repo is renamed.

## Target Structure

```
hyper-glyphy/
├── languages/
│   └── sumerian/
│       ├── scripts/                   # numbered pipeline (01-10), normalize, audit, coverage, etc.
│       │   ├── analysis/              # sumerian_anomaly_atlas, cosmogony_*, etcsl_passage_finder, preflight
│       │   └── docs/                  # consistency.py
│       ├── tests/
│       │   └── analysis/
│       ├── final_output/              # sumerian_lookup.py + gitignored binaries
│       ├── data/                      # raw/, processed/, dictionaries/ (gitignored)
│       ├── models/                    # fasttext_sumerian.*, ridge_weights_* (gitignored)
│       ├── results/                   # alignment results, audits, concept clusters
│       └── docs/                      # cosmogony, anomaly findings, EXPERIMENT_JOURNAL, NEAR_TERM_STRATEGY
├── framework/
│   └── analysis/                      # anomaly_framework, anomaly_lenses, english_displacement, semantic_field, umap_projection
├── shared/
│   ├── scripts/                       # embed_english_gemma, whiten_gemma, download_glove
│   ├── tests/                         # test_embed_english_gemma
│   └── models/                        # english_gemma_*.npz, whitening transforms (gitignored)
├── docs/                              # ROADMAP, RESEARCH_VISION, superpowers/
├── README.md
├── requirements.txt
├── pytest.ini
└── .gitignore
```

## File Mapping

### → `languages/sumerian/scripts/`

- `scripts/01_scrape_etcsl.py` through `scripts/10_export_production.py`
- `scripts/scrape_etcsl_01.py`, `scrape_cdli_02.py`, `scrape_oracc_03.py`, `dedup_04.py`, `clean_05.py`, `anchors_06.py`, `fasttext_07.py`, `fuse_08.py`, `align_09.py`, `align_09b.py`, `export_10.py`
- `scripts/sumerian_normalize.py`
- `scripts/audit_anchors.py`
- `scripts/coverage_diagnostic.py`
- `scripts/ridge_alpha_sweep.py`
- `scripts/validate_phase_b.py`
- `scripts/__init__.py`

### → `languages/sumerian/scripts/analysis/`

- `scripts/analysis/sumerian_anomaly_atlas.py`
- `scripts/analysis/cosmogony_concepts.py`
- `scripts/analysis/generate_cosmogony_tables.py`
- `scripts/analysis/generate_cosmogony_figures.py`
- `scripts/analysis/etcsl_passage_finder.py`
- `scripts/analysis/preflight_concept_check.py`

### → `languages/sumerian/scripts/docs/`

- `scripts/docs/consistency.py`
- `scripts/docs/__init__.py`

### → `languages/sumerian/tests/`

- All `tests/test_*.py` except `test_embed_english_gemma.py`

### → `languages/sumerian/tests/analysis/`

- `tests/analysis/test_anomaly_lenses.py`
- `tests/analysis/test_semantic_field.py`
- `tests/analysis/test_umap_projection.py`
- `tests/analysis/test_preflight_concept_check.py`
- `tests/analysis/test_english_displacement.py`
- `tests/analysis/test_etcsl_passage_finder.py`
- `tests/analysis/__init__.py`

### → `languages/sumerian/final_output/`

- `final_output/sumerian_lookup.py`
- `final_output/__init__.py`

### → `languages/sumerian/data/`, `models/`, `results/`, `docs/`

- `data/` → `languages/sumerian/data/` (as-is)
- `models/fasttext_sumerian.*`, `models/fused_embeddings_1536d.npz`, `models/ridge_weights*.npz` → `languages/sumerian/models/`
- `results/` → `languages/sumerian/results/` (as-is)
- Sumerian-specific docs: `anomaly_atlas_findings.*`, `anomaly_atlas.json`, `anomalies/`, `sumerian_cosmogony.md`, `cosmogony_tables.json`, `figures/`, `fonts/`, `templates/`, `EXPERIMENT_JOURNAL.md`, `NEAR_TERM_STRATEGY.md` → `languages/sumerian/docs/`

### → `framework/analysis/`

- `scripts/analysis/anomaly_framework.py`
- `scripts/analysis/anomaly_lenses.py`
- `scripts/analysis/english_displacement.py`
- `scripts/analysis/semantic_field.py`
- `scripts/analysis/umap_projection.py`
- `scripts/analysis/__init__.py`

### → `shared/scripts/`

- `scripts/embed_english_gemma.py`
- `scripts/whiten_gemma.py`
- `scripts/download_glove.py`

### → `shared/tests/`

- `tests/test_embed_english_gemma.py`

### → `shared/models/`

- `models/english_gemma_*.npz`, `models/gemma_*_whitening_transform.npz`

### → top-level `docs/` (repo-wide)

- `docs/ROADMAP.md`
- `docs/RESEARCH_VISION.md`
- `docs/superpowers/`

### Stays at top level

- `README.md`, `requirements.txt`, `pytest.ini`, `.gitignore`, `assets/`

## Import Rewriting Rules

| Old pattern | New pattern |
|---|---|
| `from scripts.<sumerian_module>` | `from languages.sumerian.scripts.<sumerian_module>` |
| `from scripts.analysis.anomaly_framework` | `from framework.analysis.anomaly_framework` |
| `from scripts.analysis.anomaly_lenses` | `from framework.analysis.anomaly_lenses` |
| `from scripts.analysis.english_displacement` | `from framework.analysis.english_displacement` |
| `from scripts.analysis.semantic_field` | `from framework.analysis.semantic_field` |
| `from scripts.analysis.umap_projection` | `from framework.analysis.umap_projection` |
| `from scripts.analysis.<sumerian_module>` | `from languages.sumerian.scripts.analysis.<sumerian_module>` |
| `from scripts.docs.<module>` | `from languages.sumerian.scripts.docs.<module>` |
| `from final_output.sumerian_lookup` | `from languages.sumerian.final_output.sumerian_lookup` |

## Other Updates

- **`sys.path` hacks**: adjust parent-traversal depth in files that compute `_ROOT` from `__file__`
- **`pytest.ini`**: `testpaths = languages/sumerian/tests shared/tests framework`
- **`__init__.py` files**: create at every new package directory
- **`.gitignore`**: update paths to reflect new locations
- **`README.md`**: update paths, usage examples, structure diagram
- **GitHub rename**: `gh repo rename hyper-glyphy` after all code changes land

## Decisions

- No `comparative/` directory until Phase β needs it (YAGNI)
- Sumerian-specific docs live in `languages/sumerian/docs/` (self-contained language units)
- Framework analysis code nests under `framework/analysis/` (mirrors old `scripts/analysis/` convention)
- Shared English-target scripts live in `shared/scripts/` now (language-independent)
- Tests co-located with each area (`languages/sumerian/tests/`, `shared/tests/`, etc.)
- Big-bang approach: one commit for structural moves, one for import fixes

## Verification

- All 167 tests pass after import rewriting
- `python -c "from languages.sumerian.final_output.sumerian_lookup import SumerianLookup"` works
- No stale `scripts/`, `tests/`, or `final_output/` directories remain at top level
- GitHub repo successfully renamed
