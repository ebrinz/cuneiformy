# Hyper-glyphy Monorepo Reorganization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure `cuneiformy` into `hyper-glyphy` — a monorepo hosting multiple ancient-language embedding spaces as siblings under one framework.

**Architecture:** Big-bang approach in two commits. Commit 1: all `git mv` structural moves + new `__init__.py` files. Commit 2: rewrite every internal import, fix `sys.path` hacks, update `pytest.ini`, `.gitignore`, `README.md`. Then rename the GitHub repo.

**Tech Stack:** Python, pytest, git, GitHub CLI (`gh`)

**Spec:** `docs/superpowers/specs/2026-05-03-hyper-glyphy-monorepo-reorg-design.md`

---

## File Structure (post-reorg)

```
hyper-glyphy/
├── languages/
│   └── sumerian/
│       ├── scripts/           # 01-10 pipeline, aliases, normalize, audit, coverage, ridge_alpha_sweep, validate_phase_b, evaluate_concept_clusters
│       │   ├── analysis/      # sumerian_anomaly_atlas, cosmogony_*, etcsl_passage_finder, preflight
│       │   └── docs/          # consistency.py
│       ├── tests/             # test_01-10, test_audit, test_coverage, test_normalize, test_integration, test_anomaly_findings_consistency
│       │   └── analysis/      # test_anomaly_lenses, test_semantic_field, test_umap, test_preflight, test_english_displacement, test_etcsl_passage_finder
│       ├── final_output/      # sumerian_lookup.py, __init__.py
│       ├── data/              # raw/, processed/, dictionaries/
│       ├── models/            # fasttext_sumerian.*, fused_*, ridge_weights*
│       ├── results/           # alignment_results, audits, clusters
│       ├── docs/              # cosmogony, anomaly findings, EXPERIMENT_JOURNAL, NEAR_TERM_STRATEGY, anomalies/, figures/, fonts/, templates/
│       └── assets/            # (empty for now, banner stays top-level)
├── framework/
│   └── analysis/              # anomaly_framework, anomaly_lenses, english_displacement, semantic_field, umap_projection
├── shared/
│   ├── scripts/               # embed_english_gemma, whiten_gemma, download_glove
│   ├── tests/                 # test_embed_english_gemma
│   └── models/                # english_gemma_*.npz, gemma_*whitening_transform.npz
├── docs/                      # ROADMAP, RESEARCH_VISION, superpowers/
├── README.md
├── requirements.txt
├── pytest.ini
└── .gitignore
```

---

### Task 1: Create destination directories and `__init__.py` files

**Files:**
- Create: `languages/__init__.py`
- Create: `languages/sumerian/__init__.py`
- Create: `languages/sumerian/scripts/__init__.py` (will be replaced by git mv in Task 2)
- Create: `languages/sumerian/scripts/analysis/__init__.py` (will be replaced by git mv)
- Create: `languages/sumerian/scripts/docs/__init__.py` (will be replaced by git mv)
- Create: `languages/sumerian/tests/__init__.py`
- Create: `languages/sumerian/tests/analysis/__init__.py` (will be replaced by git mv)
- Create: `languages/sumerian/final_output/__init__.py` (will be replaced by git mv)
- Create: `framework/__init__.py`
- Create: `framework/analysis/__init__.py` (will be replaced by git mv)
- Create: `shared/__init__.py`
- Create: `shared/scripts/__init__.py`
- Create: `shared/tests/__init__.py`

- [ ] **Step 1: Create all package directories and `__init__.py` files**

```bash
mkdir -p languages/sumerian/scripts/analysis
mkdir -p languages/sumerian/scripts/docs
mkdir -p languages/sumerian/tests/analysis
mkdir -p languages/sumerian/final_output
mkdir -p languages/sumerian/data
mkdir -p languages/sumerian/models
mkdir -p languages/sumerian/results
mkdir -p languages/sumerian/docs
mkdir -p framework/analysis
mkdir -p shared/scripts
mkdir -p shared/tests
mkdir -p shared/models

touch languages/__init__.py
touch languages/sumerian/__init__.py
touch languages/sumerian/tests/__init__.py
touch shared/__init__.py
touch shared/scripts/__init__.py
touch shared/tests/__init__.py
touch framework/__init__.py
```

Note: `languages/sumerian/scripts/__init__.py`, `languages/sumerian/scripts/analysis/__init__.py`, `languages/sumerian/scripts/docs/__init__.py`, `languages/sumerian/tests/analysis/__init__.py`, `languages/sumerian/final_output/__init__.py`, and `framework/analysis/__init__.py` will be created by `git mv` of existing `__init__.py` files in Task 2. Do NOT create them here — git mv will fail if the destination exists.

---

### Task 2: Structural moves — `git mv` everything

All moves in one batch. The order doesn't matter since it's all one commit, but grouping by destination makes it auditable.

**Important:** The shorthand alias scripts (e.g. `scripts/align_09.py`) use `os.path.join(os.path.dirname(__file__), "09_align_and_evaluate.py")` — they load their sibling by relative path from `__file__`. Since both the alias and its target move together into the same directory, these continue to work without changes.

- [ ] **Step 1: Move Sumerian pipeline scripts**

```bash
# Core pipeline + aliases
git mv scripts/__init__.py languages/sumerian/scripts/__init__.py
git mv scripts/01_scrape_etcsl.py languages/sumerian/scripts/
git mv scripts/02_scrape_cdli.py languages/sumerian/scripts/
git mv scripts/03_scrape_oracc.py languages/sumerian/scripts/
git mv scripts/04_deduplicate_corpus.py languages/sumerian/scripts/
git mv scripts/05_clean_and_tokenize.py languages/sumerian/scripts/
git mv scripts/06_extract_anchors.py languages/sumerian/scripts/
git mv scripts/07_train_fasttext.py languages/sumerian/scripts/
git mv scripts/08_fuse_embeddings.py languages/sumerian/scripts/
git mv scripts/09_align_and_evaluate.py languages/sumerian/scripts/
git mv scripts/09b_align_gemma.py languages/sumerian/scripts/
git mv scripts/10_export_production.py languages/sumerian/scripts/
git mv scripts/scrape_etcsl_01.py languages/sumerian/scripts/
git mv scripts/scrape_cdli_02.py languages/sumerian/scripts/
git mv scripts/scrape_oracc_03.py languages/sumerian/scripts/
git mv scripts/dedup_04.py languages/sumerian/scripts/
git mv scripts/clean_05.py languages/sumerian/scripts/
git mv scripts/anchors_06.py languages/sumerian/scripts/
git mv scripts/fasttext_07.py languages/sumerian/scripts/
git mv scripts/fuse_08.py languages/sumerian/scripts/
git mv scripts/align_09.py languages/sumerian/scripts/
git mv scripts/align_09b.py languages/sumerian/scripts/
git mv scripts/export_10.py languages/sumerian/scripts/

# Sumerian-specific support scripts
git mv scripts/sumerian_normalize.py languages/sumerian/scripts/
git mv scripts/audit_anchors.py languages/sumerian/scripts/
git mv scripts/coverage_diagnostic.py languages/sumerian/scripts/
git mv scripts/ridge_alpha_sweep.py languages/sumerian/scripts/
git mv scripts/validate_phase_b.py languages/sumerian/scripts/
git mv scripts/evaluate_concept_clusters.py languages/sumerian/scripts/
```

- [ ] **Step 2: Move Sumerian analysis scripts**

```bash
git mv scripts/analysis/__init__.py languages/sumerian/scripts/analysis/__init__.py
git mv scripts/analysis/sumerian_anomaly_atlas.py languages/sumerian/scripts/analysis/
git mv scripts/analysis/cosmogony_concepts.py languages/sumerian/scripts/analysis/
git mv scripts/analysis/generate_cosmogony_tables.py languages/sumerian/scripts/analysis/
git mv scripts/analysis/generate_cosmogony_figures.py languages/sumerian/scripts/analysis/
git mv scripts/analysis/etcsl_passage_finder.py languages/sumerian/scripts/analysis/
git mv scripts/analysis/preflight_concept_check.py languages/sumerian/scripts/analysis/
```

- [ ] **Step 3: Move Sumerian docs scripts**

```bash
git mv scripts/docs/__init__.py languages/sumerian/scripts/docs/__init__.py
git mv scripts/docs/consistency.py languages/sumerian/scripts/docs/
```

- [ ] **Step 4: Move framework analysis modules**

```bash
# Note: scripts/analysis/__init__.py was already moved in Step 2.
# framework/analysis/ needs its own __init__.py — create it.
git mv scripts/analysis/anomaly_framework.py framework/analysis/
git mv scripts/analysis/anomaly_lenses.py framework/analysis/
git mv scripts/analysis/english_displacement.py framework/analysis/
git mv scripts/analysis/semantic_field.py framework/analysis/
git mv scripts/analysis/umap_projection.py framework/analysis/
touch framework/analysis/__init__.py
```

- [ ] **Step 5: Move shared English-target scripts**

```bash
git mv scripts/embed_english_gemma.py shared/scripts/
git mv scripts/whiten_gemma.py shared/scripts/
git mv scripts/download_glove.py shared/scripts/
```

- [ ] **Step 6: Move Sumerian tests**

```bash
git mv tests/test_01_scrape_etcsl.py languages/sumerian/tests/
git mv tests/test_02_scrape_cdli.py languages/sumerian/tests/
git mv tests/test_03_scrape_oracc.py languages/sumerian/tests/
git mv tests/test_04_deduplicate.py languages/sumerian/tests/
git mv tests/test_05_clean.py languages/sumerian/tests/
git mv tests/test_06_anchors.py languages/sumerian/tests/
git mv tests/test_07_fasttext.py languages/sumerian/tests/
git mv tests/test_08_fuse.py languages/sumerian/tests/
git mv tests/test_09_align.py languages/sumerian/tests/
git mv tests/test_09b_align.py languages/sumerian/tests/
git mv tests/test_10_export.py languages/sumerian/tests/
git mv tests/test_audit_anchors.py languages/sumerian/tests/
git mv tests/test_coverage_diagnostic.py languages/sumerian/tests/
git mv tests/test_sumerian_normalize.py languages/sumerian/tests/
git mv tests/test_integration.py languages/sumerian/tests/
git mv tests/test_anomaly_findings_consistency.py languages/sumerian/tests/
```

- [ ] **Step 7: Move Sumerian analysis tests**

```bash
git mv tests/analysis/__init__.py languages/sumerian/tests/analysis/__init__.py
git mv tests/analysis/test_anomaly_lenses.py languages/sumerian/tests/analysis/
git mv tests/analysis/test_english_displacement.py languages/sumerian/tests/analysis/
git mv tests/analysis/test_etcsl_passage_finder.py languages/sumerian/tests/analysis/
git mv tests/analysis/test_preflight_concept_check.py languages/sumerian/tests/analysis/
git mv tests/analysis/test_semantic_field.py languages/sumerian/tests/analysis/
git mv tests/analysis/test_umap_projection.py languages/sumerian/tests/analysis/
```

- [ ] **Step 8: Move shared test**

```bash
git mv tests/test_embed_english_gemma.py shared/tests/
```

- [ ] **Step 9: Move final_output**

```bash
git mv final_output/__init__.py languages/sumerian/final_output/__init__.py
git mv final_output/sumerian_lookup.py languages/sumerian/final_output/
```

Note: The gitignored binary files (`*.npz`, `*.pkl`) in `final_output/` won't be tracked by git. Manually move them:

```bash
mv final_output/*.npz languages/sumerian/final_output/ 2>/dev/null
mv final_output/*.pkl languages/sumerian/final_output/ 2>/dev/null
```

- [ ] **Step 10: Move data, models, results**

These are mostly gitignored. Use plain `mv` for untracked content, `git mv` for tracked content.

```bash
# data/ — all gitignored content; GloVe stays here (7+ Sumerian scripts read it directly)
mv data/raw languages/sumerian/data/ 2>/dev/null
mv data/processed languages/sumerian/data/ 2>/dev/null
mv data/dictionaries languages/sumerian/data/ 2>/dev/null

# models/ — all gitignored; split between sumerian and shared
# Shared (English-target artifacts):
mv models/english_gemma_768d.npz shared/models/ 2>/dev/null
mv models/english_gemma_bare_768d.npz shared/models/ 2>/dev/null
mv models/english_gemma_bare_whitened_768d.npz shared/models/ 2>/dev/null
mv models/english_gemma_whitened_768d.npz shared/models/ 2>/dev/null
mv models/gemma_bare_whitening_transform.npz shared/models/ 2>/dev/null
mv models/gemma_whitening_transform.npz shared/models/ 2>/dev/null
# Sumerian-specific:
mv models/fasttext_sumerian.* languages/sumerian/models/ 2>/dev/null
mv models/fused_embeddings_1536d.npz languages/sumerian/models/ 2>/dev/null
mv models/ridge_weights*.npz languages/sumerian/models/ 2>/dev/null

# results/ — some tracked JSON/MD files
git mv results/* languages/sumerian/results/
```

- [ ] **Step 11: Move Sumerian docs**

```bash
git mv docs/anomaly_atlas_findings.md languages/sumerian/docs/
git mv docs/anomaly_atlas_findings.pdf languages/sumerian/docs/
git mv docs/anomaly_atlas.json languages/sumerian/docs/
git mv docs/anomalies languages/sumerian/docs/
git mv docs/sumerian_cosmogony.md languages/sumerian/docs/
git mv docs/cosmogony_tables.json languages/sumerian/docs/
git mv docs/figures languages/sumerian/docs/
git mv docs/fonts languages/sumerian/docs/
git mv docs/templates languages/sumerian/docs/
git mv docs/EXPERIMENT_JOURNAL.md languages/sumerian/docs/
git mv docs/NEAR_TERM_STRATEGY.md languages/sumerian/docs/
```

Repo-wide docs stay: `docs/ROADMAP.md`, `docs/RESEARCH_VISION.md`, `docs/superpowers/`.

- [ ] **Step 12: Clean up empty old directories**

```bash
# Remove now-empty directories (only if truly empty)
rmdir scripts/analysis scripts/docs scripts 2>/dev/null
rmdir tests/analysis tests 2>/dev/null
rmdir final_output 2>/dev/null
rmdir data/raw data/processed data/dictionaries data 2>/dev/null
rmdir models 2>/dev/null
rmdir results 2>/dev/null
```

- [ ] **Step 13: Verify the structural move**

```bash
# No Python files should remain in old locations
find scripts tests/test_*.py final_output -name "*.py" 2>/dev/null
# Expected: no output (directories should not exist)

# All Python files should be in new locations
find languages framework shared -name "*.py" | wc -l
# Expected: ~65 files
```

- [ ] **Step 14: Commit structural moves**

```bash
git add -A
git commit -m "refactor: restructure into hyper-glyphy monorepo layout

Move Sumerian-specific code under languages/sumerian/,
civilization-agnostic analysis framework under framework/,
shared English-target artifacts under shared/.

No import changes yet — next commit fixes all references."
```

---

### Task 3: Rewrite all internal imports

Every `from scripts.X` and `from final_output.X` import must be updated. This task is purely mechanical text replacement.

**Files to modify** (complete list with exact line numbers from exploration):

#### 3A: Sumerian scripts that import other Sumerian modules

- [ ] **Step 1: Fix `languages/sumerian/scripts/06_extract_anchors.py`**

Line 15-17 — `_ROOT` was `Path(__file__).parent.parent` (scripts → repo root). Now the file is at `languages/sumerian/scripts/`, so root is `.parent.parent.parent.parent`.

```python
# Old (line 15-17):
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# New:
_ROOT = Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
```

Line 19 — import:

```python
# Old:
from scripts.sumerian_normalize import normalize_sumerian_token  # noqa: E402

# New:
from languages.sumerian.scripts.sumerian_normalize import normalize_sumerian_token  # noqa: E402
```

Lines 21-23 — data paths were `Path(__file__).parent.parent / "data"`. Now: `Path(__file__).parent.parent / "data"` still works because `__file__` is in `languages/sumerian/scripts/` and `data` is at `languages/sumerian/data/` — that's `.parent` (= `languages/sumerian/`). **Wait — `.parent.parent` would go to `languages/`.** Fix:

```python
# Old (lines 21-23):
DATA_RAW = Path(__file__).parent.parent / "data" / "raw"
DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"
DATA_DICTS = Path(__file__).parent.parent / "data" / "dictionaries"

# New — .parent goes from scripts/ to sumerian/:
DATA_RAW = Path(__file__).parent.parent / "data" / "raw"
DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"
DATA_DICTS = Path(__file__).parent.parent / "data" / "dictionaries"
```

Actually these are CORRECT as-is. The old layout: `scripts/06_extract_anchors.py` → `.parent` = `scripts/` → `.parent` = repo root → `/ "data"`. New layout: `languages/sumerian/scripts/06_extract_anchors.py` → `.parent` = `languages/sumerian/scripts/` → `.parent` = `languages/sumerian/` → `/ "data"`. Since `data/` now lives at `languages/sumerian/data/`, `.parent.parent` still resolves correctly. **No change needed for data paths in Sumerian scripts.**

This pattern applies to ALL numbered pipeline scripts (01-10) and support scripts. The `Path(__file__).parent.parent / "data"` and `Path(__file__).parent.parent / "models"` patterns all resolve correctly post-move because data/models/results move to the same relative position under `languages/sumerian/`.

- [ ] **Step 2: Fix `languages/sumerian/scripts/09b_align_gemma.py`**

Lines 16-18 — `_ROOT` sys.path hack:

```python
# Old:
_ROOT = Path(__file__).parent.parent
# New:
_ROOT = Path(__file__).parent.parent.parent.parent
```

Line 23 — import:

```python
# Old:
from scripts.align_09 import (
# New:
from languages.sumerian.scripts.align_09 import (
```

Line 29 — `ROOT` for data paths:

```python
# Old:
ROOT = Path(__file__).parent.parent
# New (data is at languages/sumerian/, which is .parent.parent from scripts/):
ROOT = Path(__file__).parent.parent
```

This is correct as-is (same reasoning as Step 1).

- [ ] **Step 3: Fix `languages/sumerian/scripts/coverage_diagnostic.py`**

Lines 24-26 — `_ROOT` sys.path hack:

```python
# Old:
_ROOT = Path(__file__).parent.parent
# New:
_ROOT = Path(__file__).parent.parent.parent.parent
```

Lines 31-37 — imports:

```python
# Old:
from scripts.audit_anchors import (
# New:
from languages.sumerian.scripts.audit_anchors import (

# Old:
from scripts.sumerian_normalize import normalize_sumerian_token
# New:
from languages.sumerian.scripts.sumerian_normalize import normalize_sumerian_token
```

Line 783 — late import:

```python
# Old:
    from scripts.audit_anchors import AuditContext, classify_all
# New:
    from languages.sumerian.scripts.audit_anchors import AuditContext, classify_all
```

- [ ] **Step 4: Fix `languages/sumerian/scripts/ridge_alpha_sweep.py`**

Line 20 — import:

```python
# Old:
from scripts.align_09 import (
# New:
from languages.sumerian.scripts.align_09 import (
```

Line 26 — `ROOT`:

```python
# Old:
ROOT = Path(__file__).parent.parent
# New (resolves to languages/sumerian/ — correct):
ROOT = Path(__file__).parent.parent
```

No change needed for ROOT.

- [ ] **Step 5: Fix `languages/sumerian/scripts/validate_phase_b.py`**

Lines 19-22 — ROOT + sys.path:

```python
# Old:
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# New:
ROOT = Path(__file__).parent.parent
_REPO_ROOT = Path(__file__).parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
```

Line 53 — import:

```python
# Old:
    from final_output.sumerian_lookup import SumerianLookup
# New:
    from languages.sumerian.final_output.sumerian_lookup import SumerianLookup
```

#### 3B: Sumerian analysis scripts

- [ ] **Step 6: Fix `languages/sumerian/scripts/analysis/sumerian_anomaly_atlas.py`**

Lines 19-21 — `_ROOT` sys.path hack. Old: `.parent.parent.parent` (analysis → scripts → repo root). New: `.parent.parent.parent.parent.parent` (analysis → scripts → sumerian → languages → repo root). But `_ROOT` is also used to build AnomalyConfig paths. Those paths point to `final_output/`, `data/`, `models/`, `docs/` — which now live under `languages/sumerian/`, not repo root. So we need TWO roots:

```python
# Old:
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# New:
_REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent
_LANG_ROOT = Path(__file__).parent.parent.parent  # languages/sumerian/
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
```

Line 23 — import:

```python
# Old:
from scripts.analysis.anomaly_framework import AnomalyConfig, run_atlas
# New:
from framework.analysis.anomaly_framework import AnomalyConfig, run_atlas
```

Lines 32-53 — all `ROOT / "final_output"` etc. become `_LANG_ROOT / "final_output"`:

```python
# Old:
    ROOT = _ROOT
    config = AnomalyConfig(
        ...
        aligned_gemma_path=ROOT / "final_output" / "sumerian_aligned_gemma_vectors.npz",
        ...
        target_gemma_vocab_path=ROOT / "models" / "english_gemma_whitened_768d.npz",
        ...

# New:
    config = AnomalyConfig(
        ...
        aligned_gemma_path=_LANG_ROOT / "final_output" / "sumerian_aligned_gemma_vectors.npz",
        ...
        target_gemma_vocab_path=_REPO_ROOT / "shared" / "models" / "english_gemma_whitened_768d.npz",
        ...
```

Note: `target_gemma_vocab_path` points to English Gemma which is now in `shared/models/`. All other paths (`final_output/`, `data/`, `docs/`) stay relative to `_LANG_ROOT`.

- [ ] **Step 7: Fix `languages/sumerian/scripts/analysis/generate_cosmogony_tables.py`**

Lines 25-27 — `_ROOT`:

```python
# Old:
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
# New:
_REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent
_LANG_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
```

Line 36:

```python
# Old:
ROOT = _ROOT
# New:
ROOT = _LANG_ROOT
```

Lines 31-34 — imports:

```python
# Old:
from final_output.sumerian_lookup import SumerianLookup
from scripts.analysis.cosmogony_concepts import PRIMARY_CONCEPTS, ANUNNAKI_VOCABULARY
from scripts.analysis.english_displacement import english_displacement
from scripts.analysis.etcsl_passage_finder import find_passages

# New:
from languages.sumerian.final_output.sumerian_lookup import SumerianLookup
from languages.sumerian.scripts.analysis.cosmogony_concepts import PRIMARY_CONCEPTS, ANUNNAKI_VOCABULARY
from framework.analysis.english_displacement import english_displacement
from languages.sumerian.scripts.analysis.etcsl_passage_finder import find_passages
```

- [ ] **Step 8: Fix `languages/sumerian/scripts/analysis/generate_cosmogony_figures.py`**

Lines 20-22 — `_ROOT`:

```python
# Old:
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
# New:
_REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent
_LANG_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
```

Line 35:

```python
# Old:
ROOT = _ROOT
# New:
ROOT = _LANG_ROOT
```

Lines 26-33 — imports:

```python
# Old:
from final_output.sumerian_lookup import SumerianLookup
from scripts.analysis.cosmogony_concepts import (
...
from scripts.analysis.semantic_field import (
...
from scripts.analysis.umap_projection import umap_cosmogonic_vocabulary

# New:
from languages.sumerian.final_output.sumerian_lookup import SumerianLookup
from languages.sumerian.scripts.analysis.cosmogony_concepts import (
...
from framework.analysis.semantic_field import (
...
from framework.analysis.umap_projection import umap_cosmogonic_vocabulary
```

- [ ] **Step 9: Fix `languages/sumerian/scripts/analysis/etcsl_passage_finder.py`**

Line 14 — import:

```python
# Old:
from scripts.sumerian_normalize import normalize_sumerian_token
# New:
from languages.sumerian.scripts.sumerian_normalize import normalize_sumerian_token
```

- [ ] **Step 10: Fix `languages/sumerian/scripts/analysis/preflight_concept_check.py`**

Line 20 — import:

```python
# Old:
from scripts.sumerian_normalize import normalize_sumerian_token
# New:
from languages.sumerian.scripts.sumerian_normalize import normalize_sumerian_token
```

#### 3C: Framework module (internal import)

- [ ] **Step 11: Fix `framework/analysis/anomaly_framework.py`**

Line 260 — import:

```python
# Old:
from scripts.analysis.anomaly_lenses import (
# New:
from framework.analysis.anomaly_lenses import (
```

#### 3D: Sumerian tests — pipeline tests

- [ ] **Step 12: Fix all Sumerian pipeline test imports**

Each test file has inline imports like `from scripts.X import Y`. Replace `scripts.` with `languages.sumerian.scripts.` in these files:

**`languages/sumerian/tests/test_01_scrape_etcsl.py`** — lines 46, 58, 72, 86:
```python
# Old: from scripts.scrape_etcsl_01 import ...
# New: from languages.sumerian.scripts.scrape_etcsl_01 import ...
```

**`languages/sumerian/tests/test_02_scrape_cdli.py`** — lines 39, 50, 62, 70:
```python
# Old: from scripts.scrape_cdli_02 import ...
# New: from languages.sumerian.scripts.scrape_cdli_02 import ...
```

**`languages/sumerian/tests/test_03_scrape_oracc.py`** — lines 60, 73, 84:
```python
# Old: from scripts.scrape_oracc_03 import ...
# New: from languages.sumerian.scripts.scrape_oracc_03 import ...
```

**`languages/sumerian/tests/test_04_deduplicate.py`** — lines 6, 24, 37:
```python
# Old: from scripts.dedup_04 import ...
# New: from languages.sumerian.scripts.dedup_04 import ...
```

**`languages/sumerian/tests/test_05_clean.py`** — lines 6, 28, 37, 47, 54, 62:
```python
# Old: from scripts.clean_05 import ...
# New: from languages.sumerian.scripts.clean_05 import ...
```

**`languages/sumerian/tests/test_06_anchors.py`** — lines 6, 27, 45, 61:
```python
# Old: from scripts.anchors_06 import ...
# New: from languages.sumerian.scripts.anchors_06 import ...
```

Line 87 — path computation:
```python
# Old:
    root = Path(__file__).parent.parent
# New (tests/ → sumerian/ is .parent.parent):
    root = Path(__file__).parent.parent
```
This is correct as-is — `tests/` is directly under `languages/sumerian/`, and `data/` is a sibling.

**`languages/sumerian/tests/test_07_fasttext.py`** — lines 8, 27:
```python
# Old: from scripts.fasttext_07 import ...
# New: from languages.sumerian.scripts.fasttext_07 import ...
```

**`languages/sumerian/tests/test_08_fuse.py`** — lines 7, 22:
```python
# Old: from scripts.fuse_08 import ...
# New: from languages.sumerian.scripts.fuse_08 import ...
```

**`languages/sumerian/tests/test_09_align.py`** — lines 7, 32, 59:
```python
# Old: from scripts.align_09 import ...
# New: from languages.sumerian.scripts.align_09 import ...
```

**`languages/sumerian/tests/test_09b_align.py`** — line 6:
```python
# Old: from scripts.align_09b import ...
# New: from languages.sumerian.scripts.align_09b import ...
```

**`languages/sumerian/tests/test_10_export.py`** — lines 20, 132:
```python
# Old: from final_output.sumerian_lookup import SumerianLookup
# New: from languages.sumerian.final_output.sumerian_lookup import SumerianLookup

# Old: from scripts.export_10 import project_all_vectors
# New: from languages.sumerian.scripts.export_10 import project_all_vectors
```

**`languages/sumerian/tests/test_audit_anchors.py`** — lines 18, 30, 45, 56, 67, 83, 93, 105, 115, 125, 135, 145, 162, 174, 216, 240, 256, 269, 287, 298, 353, 369, 380:
```python
# Old: from scripts.audit_anchors import ...
# New: from languages.sumerian.scripts.audit_anchors import ...
```

**`languages/sumerian/tests/test_coverage_diagnostic.py`** — lines 13, 25, 35, 52, 69, 84, 93, 132, 151, 160, 173, 186, 203, 230, 244, 257, 287, 297, 305, 320, 334, 344, 372, 390, 409, 439, 451, 468, 495, 517, 543, 625, 650, 669, 686:
```python
# Old: from scripts.coverage_diagnostic import ...
# New: from languages.sumerian.scripts.coverage_diagnostic import ...

# Old (line 151): from scripts.sumerian_normalize import ...
# New: from languages.sumerian.scripts.sumerian_normalize import ...
```

**`languages/sumerian/tests/test_sumerian_normalize.py`** — lines 5, 11, 18, 32, 39, 45, 51, 57, 65:
```python
# Old: from scripts.sumerian_normalize import ...
# New: from languages.sumerian.scripts.sumerian_normalize import ...
```

**`languages/sumerian/tests/test_integration.py`** — lines 14-19:
```python
# Old:
    from scripts.clean_05 import clean_atf_line, build_corpus
    from scripts.anchors_06 import extract_epsd2_anchors, merge_anchors
    from scripts.fasttext_07 import train_fasttext
    from scripts.fuse_08 import fuse_embeddings
    from scripts.align_09 import build_training_data, train_ridge, evaluate_alignment
    from scripts.export_10 import project_all_vectors

# New:
    from languages.sumerian.scripts.clean_05 import clean_atf_line, build_corpus
    from languages.sumerian.scripts.anchors_06 import extract_epsd2_anchors, merge_anchors
    from languages.sumerian.scripts.fasttext_07 import train_fasttext
    from languages.sumerian.scripts.fuse_08 import fuse_embeddings
    from languages.sumerian.scripts.align_09 import build_training_data, train_ridge, evaluate_alignment
    from languages.sumerian.scripts.export_10 import project_all_vectors
```

**`languages/sumerian/tests/test_anomaly_findings_consistency.py`** — lines 19, 39, 58, 80, 90, 111:
```python
# Old: from scripts.docs.consistency import ...
# New: from languages.sumerian.scripts.docs.consistency import ...
```

- [ ] **Step 13: Fix Sumerian analysis test imports**

**`languages/sumerian/tests/analysis/test_anomaly_lenses.py`** — lines 11, 43, 67, 93, 113, 130, 153, 174, 204, 222, 243, 262, 283, 359, 367, 376, 386:

Framework imports:
```python
# Old: from scripts.analysis.anomaly_framework import ...
# New: from framework.analysis.anomaly_framework import ...

# Old: from scripts.analysis.anomaly_lenses import ...
# New: from framework.analysis.anomaly_lenses import ...
```

**`languages/sumerian/tests/analysis/test_english_displacement.py`** — lines 28, 43, 56, 67, 78:
```python
# Old: from scripts.analysis.english_displacement import ...
# New: from framework.analysis.english_displacement import ...
```

**`languages/sumerian/tests/analysis/test_etcsl_passage_finder.py`** — lines 5, 26, 44, 56, 73:
```python
# Old: from scripts.analysis.etcsl_passage_finder import ...
# New: from languages.sumerian.scripts.analysis.etcsl_passage_finder import ...
```

**`languages/sumerian/tests/analysis/test_preflight_concept_check.py`** — lines 33, 51, 63, 85, 98, 115:
```python
# Old: from scripts.analysis.preflight_concept_check import ...
# New: from languages.sumerian.scripts.analysis.preflight_concept_check import ...
```

**`languages/sumerian/tests/analysis/test_semantic_field.py`** — lines 25, 36, 47, 58, 69:
```python
# Old: from scripts.analysis.semantic_field import ...
# New: from framework.analysis.semantic_field import ...
```

**`languages/sumerian/tests/analysis/test_umap_projection.py`** — lines 23, 39, 54, 72:
```python
# Old: from scripts.analysis.umap_projection import ...
# New: from framework.analysis.umap_projection import ...
```

- [ ] **Step 14: Fix shared test imports**

**`shared/tests/test_embed_english_gemma.py`** — lines 5, 12, 19, 26, 35, 42, 56, 64, 77:
```python
# Old: from scripts.embed_english_gemma import ...
# New: from shared.scripts.embed_english_gemma import ...
```

#### 3E: Shared script path fixes

- [ ] **Step 15: Fix `shared/scripts/embed_english_gemma.py`**

`ROOT = Path(__file__).parent.parent` now resolves to `shared/` (was repo root). Output to `shared/models/` is correct. But `GLOVE_PATH` would resolve to `shared/data/processed/` — GloVe actually stays in the Sumerian data dir (7+ Sumerian scripts read it directly). Fix:

Lines 23-26:

```python
# Old:
ROOT = Path(__file__).parent.parent
GLOVE_PATH = ROOT / "data" / "processed" / "glove.6B.300d.txt"
GLOSS_OUTPUT_PATH = ROOT / "models" / "english_gemma_768d.npz"
BARE_OUTPUT_PATH = ROOT / "models" / "english_gemma_bare_768d.npz"

# New:
ROOT = Path(__file__).parent.parent
_REPO_ROOT = ROOT.parent
GLOVE_PATH = _REPO_ROOT / "languages" / "sumerian" / "data" / "processed" / "glove.6B.300d.txt"
GLOSS_OUTPUT_PATH = ROOT / "models" / "english_gemma_768d.npz"
BARE_OUTPUT_PATH = ROOT / "models" / "english_gemma_bare_768d.npz"
```

- [ ] **Step 16: Fix `shared/scripts/whiten_gemma.py`**

`ROOT = Path(__file__).parent.parent` = `shared/`. `MODELS_DIR = ROOT / "models"` = `shared/models/`. All paths relative to MODELS_DIR. **No change needed for paths.**

Line 73 — error message references old script path:

```python
# Old:
        print(f"Run: python scripts/embed_english_gemma.py{suffix}", file=sys.stderr)
# New:
        print(f"Run: python shared/scripts/embed_english_gemma.py{suffix}", file=sys.stderr)
```

- [ ] **Step 17: Fix `shared/scripts/download_glove.py`**

`Path(__file__).parent.parent` now = `shared/`. GloVe stays in the Sumerian data dir. Fix:

Line 14:
```python
# Old:
DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"
# New:
_REPO_ROOT = Path(__file__).parent.parent.parent
DATA_PROCESSED = _REPO_ROOT / "languages" / "sumerian" / "data" / "processed"
```

Line 16 — heiroglyphy cross-reference depth changes:

```python
# Old (scripts/ was 1 level deep, so .parent.parent.parent = parent of repo root):
HEIROGLYPHY_GLOVE = Path(__file__).parent.parent.parent / "heiroglyphy" / "heiro_v5_getdata" / "data" / "processed" / "glove.6B.300d.txt"

# New (shared/scripts/ is 2 levels deep, so .parent.parent.parent = repo root, need one more .parent):
HEIROGLYPHY_GLOVE = _REPO_ROOT.parent / "heiroglyphy" / "heiro_v5_getdata" / "data" / "processed" / "glove.6B.300d.txt"
```

---

### Task 4: Update configuration files

- [ ] **Step 1: Update `pytest.ini`**

```ini
[pytest]
testpaths = languages/sumerian/tests shared/tests
python_files = test_*.py
python_functions = test_*
pythonpath = .
```

Note: `framework` has no tests of its own — framework modules are tested via `languages/sumerian/tests/analysis/test_anomaly_lenses.py` etc.

- [ ] **Step 2: Update `.gitignore`**

Replace all path-specific ignores with the new layout:

```gitignore
# Data (too large for git)
languages/*/data/raw/
languages/*/data/processed/glove.6B.*
languages/*/data/processed/cleaned_corpus.txt
languages/*/data/processed/merged_corpus.json
languages/*/data/processed/english_anchors.json
languages/*/data/dictionaries/

# Models (too large for git)
languages/*/models/
shared/models/

# Results (regenerable)
languages/*/results/

# Production vectors (too large for git)
languages/*/final_output/*.npz
languages/*/final_output/*.pkl

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
.eggs/
dist/
build/
*.egg

# Virtual environment
venv/
.venv/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
```

- [ ] **Step 3: Update `README.md`**

Update the structure diagram, usage example paths, and any hardcoded references to old paths. Key changes:

- Structure section: show new `languages/sumerian/`, `framework/`, `shared/` layout
- Usage example: `from languages.sumerian.final_output.sumerian_lookup import SumerianLookup`
- Vector paths: `languages/sumerian/final_output/sumerian_aligned_gemma_vectors.npz`
- Pipeline section: `languages/sumerian/scripts/01_scrape_etcsl.py` etc.
- English model paths: `shared/models/english_gemma_whitened_768d.npz`

- [ ] **Step 4: Commit import fixes and config updates**

```bash
git add -A
git commit -m "refactor: rewrite imports and config for hyper-glyphy layout

Update all internal imports to use languages.sumerian.scripts,
framework.analysis, and shared.scripts prefixes. Fix sys.path
hacks, pytest.ini testpaths, .gitignore, and README."
```

---

### Task 5: Verify all tests pass

- [ ] **Step 1: Run the full test suite**

```bash
python -m pytest -v 2>&1
```

Expected: all 168 tests pass.

- [ ] **Step 2: Verify import smoke test**

```bash
python -c "from languages.sumerian.final_output.sumerian_lookup import SumerianLookup; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Verify no stale directories**

```bash
# These should NOT exist
ls scripts/ tests/ final_output/ 2>&1
# Expected: "No such file or directory" for each

# These SHOULD exist
ls languages/sumerian/scripts/ framework/analysis/ shared/scripts/
```

---

### Task 6: Rename GitHub repo

- [ ] **Step 1: Rename via GitHub CLI**

```bash
gh repo rename hyper-glyphy
```

- [ ] **Step 2: Update local git remote**

```bash
git remote set-url origin "$(gh repo view --json sshUrl -q .sshUrl)"
```

- [ ] **Step 3: Verify remote**

```bash
git remote -v
```

Expected: URLs now show `hyper-glyphy` instead of `cuneiformy`.

---

### Task 7: Final verification and commit spec

- [ ] **Step 1: Run tests one final time**

```bash
python -m pytest -v 2>&1
```

- [ ] **Step 2: Update ROADMAP.md to mark Phase α as shipped**

Add to the "Shipped" section at the bottom of `docs/ROADMAP.md`:

```markdown
10. Hyper-glyphy Phase α — monorepo reorganization: `languages/sumerian/`, `framework/`, `shared/`.
```

Update item 1 status from "queued, not started" to "shipped".

- [ ] **Step 3: Commit roadmap update**

```bash
git add docs/ROADMAP.md
git commit -m "docs: mark hyper-glyphy Phase α as shipped in ROADMAP"
```
