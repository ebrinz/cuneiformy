# Coverage Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `scripts/coverage_diagnostic.py` plus tests — a diagnostic that takes the ~11,798 `sumerian_vocab_miss` anchors from the Workstream 2a audit and attributes each to one of six priority-ordered primary causes, while independently simulating the projected recovery of five candidate interventions (two of which get Tier-2 semantic validation via ridge projection). Then run the diagnostic and commit the baseline report pair for 2026-04-19.

**Architecture:** Pure-function core (normalization, classification, simulation, rendering) with a thin `main()` that handles I/O. Reuses existing helpers from `scripts/audit_anchors.py` (AuditContext, classify_anchor, loaders) to find misses, then applies five simulators in isolation. Tier-2 semantic validation projects synthesized vectors through already-trained ridge weights to distinguish technical recoverability from semantic correctness.

**Tech Stack:** Python 3, numpy, gensim (for full FastText model load including subword n-gram buckets), stdlib json / hashlib / collections / pathlib, pytest. No new dependencies beyond what's already in `requirements.txt`.

**Reference spec:** `docs/superpowers/specs/2026-04-19-coverage-diagnostic-design.md`

---

## Before You Begin

- Current branch: `master`. Before writing any code, cut a fresh feature branch:
  ```bash
  cd /Users/crashy/Development/cuneiformy
  git checkout -b feat/coverage-diagnostic
  ```
  All task commits land on `feat/coverage-diagnostic`. The branch-finishing step happens after Task 5 via `superpowers:finishing-a-development-branch`.

- Verify input artifacts exist locally before starting Task 5 (Tasks 1–4 can be developed against synthetic data):
  ```bash
  ls -la \
    data/processed/english_anchors.json \
    data/processed/cleaned_corpus.txt \
    data/processed/glove.6B.300d.txt \
    data/raw/oracc_lemmas.json \
    models/fasttext_sumerian.model \
    models/fused_embeddings_1536d.npz \
    models/english_gemma_whitened_768d.npz \
    models/ridge_weights_gemma_whitened.npz
  ```
  If any are missing, note which and raise it before Task 5.

- `scripts/audit_anchors.py` is imported heavily by the diagnostic. Do NOT modify it — this plan only adds new code.

---

## File Structure

**New files:**
- `scripts/coverage_diagnostic.py` — main script (~650–750 lines estimated; loaders + classifier + 5 simulators + Tier-2 + rendering + main).
- `tests/test_coverage_diagnostic.py` — unit tests with synthetic inputs and a toy FastText model (~500 lines).
- `results/coverage_diagnostic_2026-04-19.md` — generated baseline report.
- `results/coverage_diagnostic_2026-04-19.json` — generated baseline report.

**Modified files:**
- `docs/EXPERIMENT_JOURNAL.md` — append Workstream 2b-pre baseline entry.

**Consumed (existing artifacts, not modified):**
- `data/processed/english_anchors.json`
- `data/processed/cleaned_corpus.txt`
- `data/processed/glove.6B.300d.txt`
- `data/raw/oracc_lemmas.json` (REQUIRED — fail fast if missing)
- `models/fasttext_sumerian.model` (REQUIRED — full FastText object)
- `models/fused_embeddings_1536d.npz`
- `models/english_gemma_whitened_768d.npz`
- `models/ridge_weights_gemma_whitened.npz`
- `scripts/audit_anchors.py` — imported for AuditContext + classify_anchor + loader helpers.

**Untouched:**
- All other pipeline scripts, `final_output/`, other tests.

---

## Task 1: Data loaders + DiagnosticContext (TDD)

**Files:**
- Create: `scripts/coverage_diagnostic.py` (partial — loaders + context only)
- Create: `tests/test_coverage_diagnostic.py`

### Setup note

This task builds the I/O layer for the diagnostic: three new loaders (corpus frequency, ORACC lemma surface map, FastText model) plus a `DiagnosticContext` dataclass that bundles all the loaded pieces. Existing audit loaders (`_load_fused_vocab`, `_load_gemma_vocab`, `_load_glove_vocab`, `_load_anchors`) are reused by import; we do NOT re-implement them.

- [ ] **Step 1: Write the failing loader tests**

Create `tests/test_coverage_diagnostic.py` with:

```python
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


# --- Loader tests ----------------------------------------------------------


def test_load_corpus_frequency_counts_tokens():
    from scripts.coverage_diagnostic import _load_corpus_frequency

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cleaned.txt"
        path.write_text("lugal dingir lugal\nlugal mu-dug\ndingir\n", encoding="utf-8")
        freq = _load_corpus_frequency(path)
        assert freq["lugal"] == 3
        assert freq["dingir"] == 2
        assert freq["mu-dug"] == 1


def test_load_corpus_frequency_empty_file():
    from scripts.coverage_diagnostic import _load_corpus_frequency

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "empty.txt"
        path.write_text("", encoding="utf-8")
        freq = _load_corpus_frequency(path)
        assert len(freq) == 0


def test_load_lemma_surface_map_builds_citation_to_surfaces():
    from scripts.coverage_diagnostic import _load_lemma_surface_map

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "lemmas.json"
        lemmas = [
            {"cf": "lugal", "form": "lugal", "gw": "king"},
            {"cf": "lugal", "form": "lugale", "gw": "king"},
            {"cf": "lugal", "form": "lugalanene", "gw": "king"},
            {"cf": "dingir", "form": "dingir", "gw": "god"},
        ]
        path.write_text(json.dumps(lemmas), encoding="utf-8")
        mapping = _load_lemma_surface_map(path)
        assert mapping["lugal"] == {"lugal", "lugale", "lugalanene"}
        assert mapping["dingir"] == {"dingir"}


def test_load_lemma_surface_map_normalizes_via_oracc_to_atf():
    from scripts.coverage_diagnostic import _load_lemma_surface_map

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "lemmas.json"
        lemmas = [
            {"cf": "šeš", "form": "šeš", "gw": "brother"},
            {"cf": "šeš", "form": "šešeš", "gw": "brother"},
        ]
        path.write_text(json.dumps(lemmas, ensure_ascii=False), encoding="utf-8")
        mapping = _load_lemma_surface_map(path)
        # ORACC 'š' -> ATF 'sz' normalization applied
        assert "szesz" in mapping
        assert "szesz" in mapping["szesz"]
        assert "szeszesz" in mapping["szesz"]


def test_load_lemma_surface_map_skips_empty_cf_or_form():
    from scripts.coverage_diagnostic import _load_lemma_surface_map

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "lemmas.json"
        lemmas = [
            {"cf": "lugal", "form": "lugal", "gw": "king"},
            {"cf": "", "form": "lugal", "gw": "king"},
            {"cf": "lugal", "form": "", "gw": "king"},
        ]
        path.write_text(json.dumps(lemmas), encoding="utf-8")
        mapping = _load_lemma_surface_map(path)
        assert mapping == {"lugal": {"lugal"}}


def test_diagnostic_context_is_frozen():
    from scripts.coverage_diagnostic import DiagnosticContext

    ctx = _make_tiny_context()
    with pytest.raises((AttributeError, Exception)):
        ctx.fused_vocab = frozenset()


def _make_tiny_context():
    """Build a DiagnosticContext over tiny synthetic inputs for downstream tests."""
    from scripts.coverage_diagnostic import DiagnosticContext

    return DiagnosticContext(
        fused_vocab=frozenset({"lugal", "dingir", "nar", "ta", "narta"}),
        glove_vocab=frozenset({"king", "god"}),
        gemma_vocab=frozenset({"king", "god"}),
        corpus_frequency={"lugal": 10, "dingir": 8, "narta": 2, "mu": 1},
        lemma_surface_map={"lugal": frozenset({"lugal", "lugale"})},
        fasttext_model=None,
        gemma_english_vocab=["king", "god"],
        gemma_english_vectors=np.eye(2, 768, dtype=np.float32),
        ridge_gemma_coef=np.zeros((768, 1536), dtype=np.float32),
        ridge_gemma_intercept=np.zeros(768, dtype=np.float32),
    )


def test_diagnostic_context_accepts_all_required_fields():
    ctx = _make_tiny_context()
    assert "lugal" in ctx.fused_vocab
    assert ctx.corpus_frequency["dingir"] == 8
    assert "lugal" in ctx.lemma_surface_map
    assert ctx.ridge_gemma_coef.shape == (768, 1536)
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
cd /Users/crashy/Development/cuneiformy
pytest tests/test_coverage_diagnostic.py -v
```
Expected: all 7 tests FAIL with `ModuleNotFoundError: No module named 'scripts.coverage_diagnostic'`.

- [ ] **Step 3: Implement the loaders + DiagnosticContext**

Create `scripts/coverage_diagnostic.py`:

```python
"""
Sumerian Anchor Coverage Diagnostic (Workstream 2b-pre).

Takes the sumerian_vocab_miss anchors from the Workstream 2a audit and:
  1. Classifier: attributes each to ONE primary cause (priority-ordered).
  2. Simulator: reports per-intervention projected recovery, with Tier-2
     semantic validation for the two inference-based interventions.

Writes a dated report pair under `results/`:
  - coverage_diagnostic_<YYYY-MM-DD>.md
  - coverage_diagnostic_<YYYY-MM-DD>.json  (schema version 1)

See: docs/superpowers/specs/2026-04-19-coverage-diagnostic-design.md
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Reuse loaders from the audit — single source of truth for vocab shape checks.
from scripts.audit_anchors import (
    _load_anchors,
    _load_fused_vocab,
    _load_gemma_vocab,
    _load_glove_vocab,
)

# --- Module constants -------------------------------------------------------

DIAGNOSTIC_SCHEMA_VERSION = 1
DEFAULT_SEED = 42
SUBWORD_OVERLAP_MIN = 0.5

# ORACC citation-form unicode -> ATF normalization (mirror of
# scripts/06_extract_anchors.py::_ORACC_TO_ATF). Kept local to avoid importing
# the leading-digit script module.
_ORACC_TO_ATF = {
    "š": "sz", "Š": "SZ",
    "ŋ": "j",  "Ŋ": "J",
    "ḫ": "h",  "Ḫ": "H",
    "ṣ": "s",  "Ṣ": "S",
    "ṭ": "t",  "Ṭ": "T",
    "ʾ": "",
}


def _normalize_oracc_to_atf(s: str) -> str:
    """Apply ORACC -> ATF letter normalization and lowercase. No subscripts here."""
    for old, new in _ORACC_TO_ATF.items():
        s = s.replace(old, new)
    return s.lower()


# --- DiagnosticContext ------------------------------------------------------


@dataclass(frozen=True)
class DiagnosticContext:
    fused_vocab: frozenset[str]
    glove_vocab: frozenset[str]
    gemma_vocab: frozenset[str]
    corpus_frequency: dict[str, int]
    lemma_surface_map: dict[str, frozenset[str]]
    fasttext_model: Any  # gensim FastText model, or None for tests that don't need it
    gemma_english_vocab: list[str]
    gemma_english_vectors: np.ndarray  # (N, 768) float32
    ridge_gemma_coef: np.ndarray       # (768, 1536) float32
    ridge_gemma_intercept: np.ndarray  # (768,) float32


# --- Loaders ----------------------------------------------------------------


def _load_corpus_frequency(path: Path) -> dict[str, int]:
    """Count token occurrences in the cleaned corpus (whitespace-split per line)."""
    freq: Counter[str] = Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            for token in line.strip().split():
                freq[token] += 1
    return dict(freq)


def _load_lemma_surface_map(path: Path) -> dict[str, frozenset[str]]:
    """Build {citation_form: {surface_forms}} from ORACC lemmas.

    Applies the same ORACC->ATF normalization used in scripts/06_extract_anchors.py
    so the surface forms match what the corpus (and thus FastText vocab) contains.
    Empty cf or form values are skipped.
    """
    import json
    with open(path, encoding="utf-8") as f:
        lemmas = json.load(f)

    surfaces_by_cf: dict[str, set[str]] = {}
    for lemma in lemmas:
        cf_raw = (lemma.get("cf") or "").strip()
        form_raw = (lemma.get("form") or "").strip()
        if not cf_raw or not form_raw:
            continue
        cf = _normalize_oracc_to_atf(cf_raw)
        form = _normalize_oracc_to_atf(form_raw)
        if not cf or not form:
            continue
        surfaces_by_cf.setdefault(cf, set()).add(form)

    # Freeze the surface sets to match the frozen DiagnosticContext contract.
    return {cf: frozenset(forms) for cf, forms in surfaces_by_cf.items()}


def _load_fasttext_model(path: Path):
    """Load the full FastText model (needed for subword n-gram access)."""
    from gensim.models import FastText
    return FastText.load(str(path))


def _load_ridge_weights(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load ridge coef + intercept from an npz produced by align_and_evaluate."""
    data = np.load(str(path))
    coef = data["coef"].astype(np.float32)
    intercept = data["intercept"].astype(np.float32)
    if coef.shape[0] != 768 or coef.shape[1] != 1536:
        raise ValueError(
            f"Ridge coef shape {coef.shape} != (768, 1536) — "
            "expected whitened-Gemma ridge weights"
        )
    if intercept.shape != (768,):
        raise ValueError(
            f"Ridge intercept shape {intercept.shape} != (768,)"
        )
    return coef, intercept


def _load_gemma_english_npz(path: Path) -> tuple[list[str], np.ndarray]:
    """Load the whitened-Gemma English cache (vocab + vectors)."""
    data = np.load(str(path))
    vocab = [str(w) for w in data["vocab"]]
    vectors = data["vectors"].astype(np.float32)
    if vectors.shape[1] != 768:
        raise ValueError(
            f"Gemma vectors dim {vectors.shape[1]} != 768 — "
            "regenerate via scripts/whiten_gemma.py"
        )
    if vectors.shape[0] != len(vocab):
        raise ValueError("Gemma vocab/vectors row count mismatch")
    return vocab, vectors
```

- [ ] **Step 4: Run tests, verify all pass**

```bash
pytest tests/test_coverage_diagnostic.py -v
```
Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/crashy/Development/cuneiformy
git add scripts/coverage_diagnostic.py tests/test_coverage_diagnostic.py
git commit -m "feat: add coverage diagnostic loaders + DiagnosticContext"
```

---

## Task 2: Classifier with 6-bucket priority attribution (TDD)

**Files:**
- Modify: `scripts/coverage_diagnostic.py` (append normalization helper + classifier)
- Modify: `tests/test_coverage_diagnostic.py` (append classifier tests)

### Setup note

The classifier evaluates each miss against checks in priority order. First match wins. Six priority levels:
1. `normalization_recoverable` — apply full normalization (subscripts → ASCII, strip `{...}`, ORACC → ATF, drop hyphens); check explicit vocab.
2. `in_corpus_below_min_count` — exact anchor form OR normalized form is in `corpus_frequency` with count in `[1, 4]`.
3. `oracc_lemma_surface_recoverable` — anchor's citation form has a surface variant in explicit vocab.
4. `morpheme_composition_recoverable` — hyphenated anchor; all morphemes are in explicit vocab.
5. `subword_inference_recoverable` — subword n-gram overlap ≥ `SUBWORD_OVERLAP_MIN`.
6. `genuinely_missing` — none of the above.

For bucket 5 the classifier needs to compute n-gram overlap. We build a `trained_ngrams` set once (union over the fused vocab) and reuse it.

- [ ] **Step 1: Append classifier tests to `tests/test_coverage_diagnostic.py`**

Append:

```python
# --- Classifier tests ------------------------------------------------------


PRIMARY_CAUSE_ORDER = (
    "normalization_recoverable",
    "in_corpus_below_min_count",
    "oracc_lemma_surface_recoverable",
    "morpheme_composition_recoverable",
    "subword_inference_recoverable",
    "genuinely_missing",
)


def _classifier_ctx(**overrides):
    """Extend _make_tiny_context with test-specific overrides."""
    from scripts.coverage_diagnostic import DiagnosticContext

    base = dict(
        fused_vocab=frozenset({"lugal", "dingir", "nar", "ta", "narta", "za3sze3la2"}),
        glove_vocab=frozenset({"king", "god"}),
        gemma_vocab=frozenset({"king", "god"}),
        corpus_frequency={"lugal": 10, "dingir": 8, "narta": 2, "rare1": 1, "rare3": 3},
        lemma_surface_map={"kasan": frozenset({"kasan", "kasane"})},
        fasttext_model=None,
        gemma_english_vocab=["king", "god"],
        gemma_english_vectors=np.eye(2, 768, dtype=np.float32),
        ridge_gemma_coef=np.zeros((768, 1536), dtype=np.float32),
        ridge_gemma_intercept=np.zeros(768, dtype=np.float32),
    )
    base.update(overrides)
    return DiagnosticContext(**base)


def test_normalize_anchor_form_handles_subscripts_braces_and_oracc():
    from scripts.coverage_diagnostic import normalize_anchor_form

    assert normalize_anchor_form("{tug₂}mug") == "tug2mug"
    assert normalize_anchor_form("za₃-sze₃-la₂") == "za3sze3la2"
    assert normalize_anchor_form("šeš") == "sz"  # 'š' -> 'sz', rest lowercased; no letters left except sz
    assert normalize_anchor_form("mu-du₃-sze₃") == "mudu3sze3"


def test_classify_miss_normalization_recoverable_wins():
    from scripts.coverage_diagnostic import classify_miss

    ctx = _classifier_ctx()
    # za₃-sze₃-la₂ normalizes to za3sze3la2 which is in fused_vocab.
    bucket = classify_miss(
        {"sumerian": "za₃-sze₃-la₂", "english": "cosmic force", "confidence": 0.9},
        ctx,
        trained_ngrams=frozenset(),
    )
    assert bucket == "normalization_recoverable"


def test_classify_miss_in_corpus_below_min_count():
    from scripts.coverage_diagnostic import classify_miss

    ctx = _classifier_ctx()
    # "rare3" is in corpus_frequency with count 3 but NOT in fused_vocab (min_count=5).
    bucket = classify_miss(
        {"sumerian": "rare3", "english": "uncommon", "confidence": 0.9},
        ctx,
        trained_ngrams=frozenset(),
    )
    assert bucket == "in_corpus_below_min_count"


def test_classify_miss_oracc_lemma_surface_recoverable():
    from scripts.coverage_diagnostic import classify_miss

    # "kasan" is in lemma_surface_map with "kasane" among its surfaces.
    # Give fused_vocab that includes "kasane" but NOT "kasan".
    ctx = _classifier_ctx(
        fused_vocab=frozenset({"lugal", "kasane"}),
        lemma_surface_map={"kasan": frozenset({"kasan", "kasane"})},
    )
    bucket = classify_miss(
        {"sumerian": "kasan", "english": "noblewoman", "confidence": 0.9},
        ctx,
        trained_ngrams=frozenset(),
    )
    assert bucket == "oracc_lemma_surface_recoverable"


def test_classify_miss_morpheme_composition_requires_all_morphemes_in_vocab():
    from scripts.coverage_diagnostic import classify_miss

    ctx = _classifier_ctx(fused_vocab=frozenset({"nar", "ta"}))
    # "nar-ta" hyphenates into ["nar", "ta"], both in vocab.
    bucket = classify_miss(
        {"sumerian": "nar-ta", "english": "musician", "confidence": 0.9},
        ctx,
        trained_ngrams=frozenset(),
    )
    assert bucket == "morpheme_composition_recoverable"

    # "nar-missing" has one morpheme out of vocab -> NOT recoverable by morpheme comp.
    # Must fall through to subword or genuinely_missing. With no trained ngrams, genuinely_missing.
    bucket2 = classify_miss(
        {"sumerian": "nar-missing", "english": "something", "confidence": 0.9},
        ctx,
        trained_ngrams=frozenset(),
    )
    assert bucket2 == "genuinely_missing"


def test_classify_miss_subword_inference_respects_threshold():
    from scripts.coverage_diagnostic import classify_miss, _ngrams

    ctx = _classifier_ctx()
    # Build a trained_ngrams set that covers >= 50% of "xyznew"'s n-grams.
    trained = _ngrams("xyznewz", 3, 6) | _ngrams("xyznew", 3, 6)
    bucket = classify_miss(
        {"sumerian": "xyznew", "english": "something", "confidence": 0.9},
        ctx,
        trained_ngrams=trained,
    )
    assert bucket == "subword_inference_recoverable"


def test_classify_miss_subword_below_threshold_falls_to_missing():
    from scripts.coverage_diagnostic import classify_miss

    ctx = _classifier_ctx()
    # Empty trained set -> overlap is 0%. Below threshold 0.5.
    bucket = classify_miss(
        {"sumerian": "completelyunknown", "english": "absent", "confidence": 0.9},
        ctx,
        trained_ngrams=frozenset(),
    )
    assert bucket == "genuinely_missing"


def test_classify_all_misses_sums_to_total():
    from scripts.coverage_diagnostic import classify_all_misses, _ngrams

    ctx = _classifier_ctx()
    misses = [
        {"sumerian": "za₃-sze₃-la₂", "english": "force", "confidence": 0.9},  # normalization
        {"sumerian": "rare3", "english": "uncommon", "confidence": 0.9},        # below_min_count
        {"sumerian": "nar-ta", "english": "musician", "confidence": 0.9},       # morpheme
        {"sumerian": "completelyunknown", "english": "absent", "confidence": 0.9},  # missing
    ]
    trained = _ngrams("", 3, 6)  # empty set
    result = classify_all_misses(misses, ctx, trained)

    total = sum(b["count"] for b in result["primary_causes"].values())
    assert total == len(misses) == 4
    assert result["primary_causes"]["normalization_recoverable"]["count"] == 1
    assert result["primary_causes"]["in_corpus_below_min_count"]["count"] == 1
    assert result["primary_causes"]["morpheme_composition_recoverable"]["count"] == 1
    assert result["primary_causes"]["genuinely_missing"]["count"] == 1
    assert result["total_misses"] == 4
    # All priority buckets present even if zero
    assert set(result["primary_causes"].keys()) == set(PRIMARY_CAUSE_ORDER)


def test_classify_all_misses_empty_input():
    from scripts.coverage_diagnostic import classify_all_misses

    ctx = _classifier_ctx()
    result = classify_all_misses([], ctx, frozenset())
    assert result["total_misses"] == 0
    for bucket in result["primary_causes"].values():
        assert bucket["count"] == 0


def test_ngrams_respects_min_n_max_n():
    from scripts.coverage_diagnostic import _ngrams

    result = _ngrams("abc", 3, 3)
    # padded "<abc>" -> 3-grams: "<ab", "abc", "bc>"
    assert result == {"<ab", "abc", "bc>"}


def test_trained_ngrams_unions_vocab():
    from scripts.coverage_diagnostic import _trained_ngrams

    trained = _trained_ngrams(["ab", "cd"], 2, 2)
    assert "<a" in trained
    assert "ab" in trained
    assert "b>" in trained
    assert "<c" in trained
    assert "cd" in trained
    assert "d>" in trained
```

- [ ] **Step 2: Run new tests, verify they fail**

```bash
pytest tests/test_coverage_diagnostic.py -v
```
Expected: 10 new classifier tests FAIL on `ImportError`. The 7 loader tests still PASS.

- [ ] **Step 3: Append classifier code to `scripts/coverage_diagnostic.py`**

Append:

```python

# --- Normalization ---------------------------------------------------------

import re  # noqa: E402

_SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def normalize_anchor_form(raw: str) -> str:
    """Mirror the normalization chain in scripts/05_clean_and_tokenize.py.

    Applies (in order):
      1. Unicode subscript digits -> ASCII digits.
      2. Strip determinative braces {...} keeping content.
      3. ORACC unicode Sumerian letters -> ATF (š -> sz, etc.).
      4. Drop hyphens (produces the fully-joined compound form).
      5. Lowercase.
    """
    s = str(raw or "")
    s = s.translate(_SUBSCRIPT_MAP)
    s = re.sub(r"\{([^}]*)\}", r"\1", s)
    s = _normalize_oracc_to_atf(s)
    s = s.replace("-", "")
    return s.strip()


# --- n-gram helpers --------------------------------------------------------


def _ngrams(word: str, min_n: int, max_n: int) -> frozenset[str]:
    """Character n-grams with FastText-style angle-bracket padding."""
    padded = f"<{word}>"
    result: set[str] = set()
    for n in range(min_n, max_n + 1):
        if n > len(padded):
            continue
        for i in range(len(padded) - n + 1):
            result.add(padded[i : i + n])
    return frozenset(result)


def _trained_ngrams(vocab, min_n: int, max_n: int) -> frozenset[str]:
    """Union of n-grams across the training vocab. Expensive once, fast per lookup."""
    out: set[str] = set()
    for word in vocab:
        out.update(_ngrams(word, min_n, max_n))
    return frozenset(out)


def _subword_overlap(anchor: str, trained_ngrams: frozenset[str], min_n: int, max_n: int) -> float:
    anchor_ngrams = _ngrams(anchor, min_n, max_n)
    if not anchor_ngrams:
        return 0.0
    return len(anchor_ngrams & trained_ngrams) / len(anchor_ngrams)


# --- Classifier ------------------------------------------------------------

PRIMARY_CAUSE_ORDER = (
    "normalization_recoverable",
    "in_corpus_below_min_count",
    "oracc_lemma_surface_recoverable",
    "morpheme_composition_recoverable",
    "subword_inference_recoverable",
    "genuinely_missing",
)


def _morphemes(anchor_raw: str) -> list[str]:
    """Split by hyphens, normalize each morpheme individually."""
    if "-" not in anchor_raw:
        return []
    parts: list[str] = []
    for piece in anchor_raw.split("-"):
        # Normalize each morpheme: subscripts -> ASCII, ORACC -> ATF, strip braces, lowercase.
        piece = piece.translate(_SUBSCRIPT_MAP)
        piece = re.sub(r"\{([^}]*)\}", r"\1", piece)
        piece = _normalize_oracc_to_atf(piece)
        piece = piece.strip()
        if piece:
            parts.append(piece)
    return parts


def classify_miss(
    anchor: dict,
    ctx: "DiagnosticContext",
    trained_ngrams: frozenset[str],
    fasttext_min_n: int = 3,
    fasttext_max_n: int = 6,
) -> str:
    """Priority-ordered primary-cause attribution for one missing anchor."""
    sumerian_raw = str(anchor.get("sumerian") or "").strip()
    english = str(anchor.get("english") or "").lower()

    # 1. normalization_recoverable
    normalized = normalize_anchor_form(sumerian_raw)
    if normalized and normalized in ctx.fused_vocab:
        return "normalization_recoverable"

    # 2. in_corpus_below_min_count: check both raw and normalized forms.
    for candidate in (sumerian_raw, normalized):
        if not candidate:
            continue
        count = ctx.corpus_frequency.get(candidate, 0)
        if 1 <= count < 5:
            return "in_corpus_below_min_count"

    # 3. oracc_lemma_surface_recoverable: citation form (normalized) -> any surface in vocab.
    if normalized in ctx.lemma_surface_map:
        for surface in ctx.lemma_surface_map[normalized]:
            if surface in ctx.fused_vocab:
                return "oracc_lemma_surface_recoverable"

    # 4. morpheme_composition_recoverable: hyphenated, all morphemes in vocab.
    morphemes = _morphemes(sumerian_raw)
    if morphemes and all(m in ctx.fused_vocab for m in morphemes):
        return "morpheme_composition_recoverable"

    # 5. subword_inference_recoverable: n-gram overlap >= threshold on NORMALIZED form.
    if normalized:
        overlap = _subword_overlap(normalized, trained_ngrams, fasttext_min_n, fasttext_max_n)
        if overlap >= SUBWORD_OVERLAP_MIN:
            return "subword_inference_recoverable"

    return "genuinely_missing"


def classify_all_misses(
    misses: list[dict],
    ctx: "DiagnosticContext",
    trained_ngrams: frozenset[str],
    fasttext_min_n: int = 3,
    fasttext_max_n: int = 6,
) -> dict:
    """Classify every miss; return totals + per-bucket rows with traces."""
    primary_causes: dict[str, list[dict]] = {name: [] for name in PRIMARY_CAUSE_ORDER}

    for anchor in misses:
        bucket = classify_miss(anchor, ctx, trained_ngrams, fasttext_min_n, fasttext_max_n)
        # Attach a lightweight trace describing the match, for the examples field.
        trace: dict[str, Any] = {}
        sumerian_raw = str(anchor.get("sumerian") or "").strip()
        normalized = normalize_anchor_form(sumerian_raw)
        if bucket == "normalization_recoverable":
            trace["normalized_form"] = normalized
        elif bucket == "in_corpus_below_min_count":
            for candidate in (sumerian_raw, normalized):
                if 1 <= ctx.corpus_frequency.get(candidate, 0) < 5:
                    trace["matched_form"] = candidate
                    trace["corpus_count"] = ctx.corpus_frequency[candidate]
                    break
        elif bucket == "oracc_lemma_surface_recoverable":
            hits = [s for s in ctx.lemma_surface_map.get(normalized, ()) if s in ctx.fused_vocab]
            trace["matched_surface_forms"] = hits[:3]
        elif bucket == "morpheme_composition_recoverable":
            trace["morphemes_in_vocab"] = _morphemes(sumerian_raw)
        elif bucket == "subword_inference_recoverable":
            overlap = _subword_overlap(normalized, trained_ngrams, fasttext_min_n, fasttext_max_n)
            trace["ngram_overlap"] = round(overlap, 4)

        enriched = dict(anchor)
        enriched["trace"] = trace
        primary_causes[bucket].append(enriched)

    total = len(misses)
    return {
        "total_misses": total,
        "primary_causes": {
            name: {
                "count": len(rows),
                "pct": (len(rows) / total * 100.0) if total else 0.0,
                "rows": rows,
            }
            for name, rows in primary_causes.items()
        },
    }
```

- [ ] **Step 4: Run tests, verify all pass**

```bash
pytest tests/test_coverage_diagnostic.py -v
```
Expected: all 17 tests PASS (7 loader + 10 classifier).

- [ ] **Step 5: Commit**

```bash
cd /Users/crashy/Development/cuneiformy
git add scripts/coverage_diagnostic.py tests/test_coverage_diagnostic.py
git commit -m "feat: add coverage diagnostic 6-bucket priority classifier"
```

---

## Task 3: Exact-match simulators (TDD)

**Files:**
- Modify: `scripts/coverage_diagnostic.py` (append 3 simulators)
- Modify: `tests/test_coverage_diagnostic.py` (append simulator tests)

### Setup note

This task adds the three exact-match simulators — those that don't require inferred vectors:
- `simulate_ascii_normalize`: apply normalization; count exact vocab hits.
- `simulate_lower_min_count`: for each min_count ∈ {1,2,3,4}, count anchors in corpus with frequency ≥ that threshold (and not already in vocab).
- `simulate_oracc_lemma_expansion`: count anchors whose citation form has an in-vocab surface variant.

Each runs independently on the full misses list (no priority-ordering interaction).

- [ ] **Step 1: Append simulator tests**

Append to `tests/test_coverage_diagnostic.py`:

```python
# --- Exact-match simulator tests -------------------------------------------


def test_simulate_ascii_normalize_counts_exact_hits():
    from scripts.coverage_diagnostic import simulate_ascii_normalize

    ctx = _classifier_ctx(fused_vocab=frozenset({"tug2mug", "za3sze3la2", "lugal"}))
    misses = [
        {"sumerian": "{tug₂}mug", "english": "garment", "confidence": 0.9},        # hits
        {"sumerian": "za₃-sze₃-la₂", "english": "force", "confidence": 0.9},       # hits
        {"sumerian": "completelyunknown", "english": "absent", "confidence": 0.9}, # misses
    ]
    result = simulate_ascii_normalize(misses, ctx)
    assert result["anchors_newly_resolvable"] == 2
    assert result["trustworthiness"] == "exact"


def test_simulate_ascii_normalize_ignores_anchors_already_normalized():
    from scripts.coverage_diagnostic import simulate_ascii_normalize

    ctx = _classifier_ctx(fused_vocab=frozenset({"dingir"}))
    # This anchor's normalized form is NOT in vocab -> not recovered by normalization.
    misses = [{"sumerian": "missing", "english": "absent", "confidence": 0.9}]
    result = simulate_ascii_normalize(misses, ctx)
    assert result["anchors_newly_resolvable"] == 0


def test_simulate_lower_min_count_per_threshold_is_monotone_nonincreasing():
    from scripts.coverage_diagnostic import simulate_lower_min_count

    ctx = _classifier_ctx(
        corpus_frequency={"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6},
        fused_vocab=frozenset({"e", "f"}),  # only >=5 made the vocab cut
    )
    misses = [
        {"sumerian": s, "english": "x", "confidence": 0.9}
        for s in ("a", "b", "c", "d", "e", "f")  # e, f already in vocab so skipped
    ]
    result = simulate_lower_min_count(misses, ctx)
    per = result["per_threshold"]
    # Anchors newly resolvable at each threshold (only miss anchors counted).
    # misses in corpus with freq >= threshold and not in vocab:
    #   t=1: a(1), b(2), c(3), d(4) -> 4
    #   t=2: b(2), c(3), d(4) -> 3
    #   t=3: c(3), d(4) -> 2
    #   t=4: d(4) -> 1
    assert per["1"]["anchors_newly_resolvable"] == 4
    assert per["2"]["anchors_newly_resolvable"] == 3
    assert per["3"]["anchors_newly_resolvable"] == 2
    assert per["4"]["anchors_newly_resolvable"] == 1
    # Monotone non-increasing across thresholds.
    counts = [per[str(t)]["anchors_newly_resolvable"] for t in (1, 2, 3, 4)]
    assert counts == sorted(counts, reverse=True)


def test_simulate_oracc_lemma_expansion_counts_unique_anchors():
    from scripts.coverage_diagnostic import simulate_oracc_lemma_expansion

    # Citation form "kasan" has three surface variants; "kasane" is in vocab.
    ctx = _classifier_ctx(
        fused_vocab=frozenset({"kasane"}),
        lemma_surface_map={"kasan": frozenset({"kasan", "kasane", "kasanene"})},
    )
    misses = [
        {"sumerian": "kasan", "english": "noblewoman", "confidence": 0.9},
        {"sumerian": "completelyunknown", "english": "absent", "confidence": 0.9},
    ]
    result = simulate_oracc_lemma_expansion(misses, ctx)
    assert result["anchors_newly_resolvable"] == 1
    # surface_forms_added_to_vocab reports total unique surface forms recovered.
    assert result["surface_forms_added_to_vocab"] >= 1


def test_simulate_oracc_lemma_expansion_requires_in_vocab_surface():
    from scripts.coverage_diagnostic import simulate_oracc_lemma_expansion

    # Citation has 2 surfaces, neither in vocab.
    ctx = _classifier_ctx(
        fused_vocab=frozenset({"unrelated"}),
        lemma_surface_map={"kasan": frozenset({"kasanx", "kasany"})},
    )
    misses = [{"sumerian": "kasan", "english": "noblewoman", "confidence": 0.9}]
    result = simulate_oracc_lemma_expansion(misses, ctx)
    assert result["anchors_newly_resolvable"] == 0
```

- [ ] **Step 2: Run new tests, verify they fail**

```bash
pytest tests/test_coverage_diagnostic.py -v
```
Expected: 5 new simulator tests FAIL on `ImportError`. Prior 17 still pass.

- [ ] **Step 3: Append exact-match simulators to `scripts/coverage_diagnostic.py`**

Append:

```python

# --- Simulators (exact-match) ----------------------------------------------


def simulate_ascii_normalize(misses: list[dict], ctx: "DiagnosticContext") -> dict:
    """Apply normalization; count anchors whose normalized form is in explicit vocab."""
    resolvable = 0
    for anchor in misses:
        normalized = normalize_anchor_form(str(anchor.get("sumerian") or ""))
        if normalized and normalized in ctx.fused_vocab:
            resolvable += 1
    return {
        "anchors_newly_resolvable": resolvable,
        "trustworthiness": "exact",
        "notes": "Pure string normalization; recovered anchors are exact vocab matches.",
    }


def simulate_lower_min_count(misses: list[dict], ctx: "DiagnosticContext") -> dict:
    """For each min_count in {1,2,3,4}, count OOV anchors with corpus freq >= that threshold."""
    per_threshold: dict[str, dict] = {}
    for threshold in (1, 2, 3, 4):
        resolvable = 0
        for anchor in misses:
            sumerian_raw = str(anchor.get("sumerian") or "").strip()
            normalized = normalize_anchor_form(sumerian_raw)
            # Skip if the form is ALREADY in explicit vocab (not a miss).
            if sumerian_raw in ctx.fused_vocab or normalized in ctx.fused_vocab:
                continue
            for candidate in (sumerian_raw, normalized):
                if not candidate:
                    continue
                if ctx.corpus_frequency.get(candidate, 0) >= threshold:
                    resolvable += 1
                    break
        per_threshold[str(threshold)] = {
            "anchors_newly_resolvable": resolvable,
        }
    return {
        "per_threshold": per_threshold,
        "trustworthiness": "exact",
        "notes": "Assumes FastText retrain; anchor form exists in cleaned_corpus.txt at the stated frequency.",
    }


def simulate_oracc_lemma_expansion(misses: list[dict], ctx: "DiagnosticContext") -> dict:
    """Count anchors whose citation form has any surface variant in explicit vocab."""
    resolvable = 0
    unique_surfaces: set[str] = set()
    for anchor in misses:
        normalized = normalize_anchor_form(str(anchor.get("sumerian") or ""))
        surfaces = ctx.lemma_surface_map.get(normalized, frozenset())
        matched = [s for s in surfaces if s in ctx.fused_vocab]
        if matched:
            resolvable += 1
            unique_surfaces.update(matched)
    return {
        "anchors_newly_resolvable": resolvable,
        "surface_forms_added_to_vocab": len(unique_surfaces),
        "trustworthiness": "exact",
        "notes": "Expansion maps each citation form to every surface variant in FastText vocab.",
    }
```

- [ ] **Step 4: Run tests, verify all pass**

```bash
pytest tests/test_coverage_diagnostic.py -v
```
Expected: all 22 tests PASS (17 prior + 5 new simulator).

- [ ] **Step 5: Commit**

```bash
cd /Users/crashy/Development/cuneiformy
git add scripts/coverage_diagnostic.py tests/test_coverage_diagnostic.py
git commit -m "feat: add coverage diagnostic exact-match simulators"
```

---

## Task 4: Inference simulators + Tier-2 semantic validation (TDD)

**Files:**
- Modify: `scripts/coverage_diagnostic.py` (append 2 inference simulators + Tier-2 helper)
- Modify: `tests/test_coverage_diagnostic.py` (append inference + Tier-2 tests)

### Setup note

This task adds the two inference-based simulators plus the Tier-2 semantic validation helper.

Tier-2 procedure for a synthesized 768d Sumerian FastText vector:
1. Fuse with zero-padding: `fused = concat([ft_vec, zeros(768)])` → 1536d (matching `scripts/08_fuse_embeddings.py`).
2. Project through ridge weights: `projected = fused @ coef.T + intercept` → 768d in whitened-Gemma space.
3. L2-normalize both `projected` and the English Gemma cache.
4. Compute cosine similarity: `sims = eng_gemma_norm @ projected_norm`.
5. Top-K nearest neighbors — check if the anchor's expected English gloss (lowercased) is among them.

The tests build tiny synthetic ridge weights + tiny English vocab so Tier-2 behavior is verifiable without loading real data.

- [ ] **Step 1: Append inference simulator tests**

Append:

```python
# --- Inference simulator + Tier-2 tests ------------------------------------


def _tier2_ctx_with_exact_identity():
    """A ctx where the ridge is identity-like (on first 768 dims) and Gemma
    English rows are canonical basis vectors, so a Sumerian input that's a
    scaled basis vector lands on the corresponding English word."""
    from scripts.coverage_diagnostic import DiagnosticContext

    # 5 English words: king, god, cow, river, mountain
    eng_vocab = ["king", "god", "cow", "river", "mountain"]
    eng_vectors = np.zeros((5, 768), dtype=np.float32)
    for i in range(5):
        eng_vectors[i, i] = 1.0

    # Ridge: coef is (768, 1536); intercept zero. coef's first 768 cols act as identity
    # on the FastText portion of the fused 1536d vector (zeros in the second half).
    coef = np.zeros((768, 1536), dtype=np.float32)
    for i in range(768):
        coef[i, i] = 1.0
    intercept = np.zeros(768, dtype=np.float32)

    return DiagnosticContext(
        fused_vocab=frozenset({"lugal", "dingir", "nar", "ta"}),
        glove_vocab=frozenset(eng_vocab),
        gemma_vocab=frozenset(eng_vocab),
        corpus_frequency={},
        lemma_surface_map={},
        fasttext_model=None,
        gemma_english_vocab=eng_vocab,
        gemma_english_vectors=eng_vectors,
        ridge_gemma_coef=coef,
        ridge_gemma_intercept=intercept,
    )


def test_tier2_project_and_nearest_neighbor_identifies_correct_english():
    from scripts.coverage_diagnostic import _tier2_nearest_english

    ctx = _tier2_ctx_with_exact_identity()
    # Synthesized FastText vec that aligns with the 'cow' basis (index 2).
    ft_vec = np.zeros(768, dtype=np.float32)
    ft_vec[2] = 1.0

    top_k = _tier2_nearest_english(ft_vec, ctx, k=3)
    assert top_k[0] == "cow"


def test_tier2_score_returns_correctness_flags():
    from scripts.coverage_diagnostic import _tier2_score_anchor

    ctx = _tier2_ctx_with_exact_identity()
    ft_vec = np.zeros(768, dtype=np.float32)
    ft_vec[3] = 1.0  # 'river'

    result = _tier2_score_anchor(ft_vec, expected_english="river", ctx=ctx)
    assert result["top1"] is True
    assert result["top5"] is True
    assert result["top10"] is True

    result2 = _tier2_score_anchor(ft_vec, expected_english="mountain", ctx=ctx)
    # "mountain" is index 4; not the top-1 hit.
    assert result2["top1"] is False


def test_simulate_morpheme_composition_tier1_counts_anchors_with_all_morphemes_in_vocab():
    from scripts.coverage_diagnostic import simulate_morpheme_composition

    ctx = _tier2_ctx_with_exact_identity()
    # Mock FastText behavior using a small dict lookup: fusion is impossible without
    # a real model. We patch the simulator's ft-vector lookup via a simple dict.
    morpheme_vectors = {
        "nar": np.zeros(768, dtype=np.float32),
        "ta": np.zeros(768, dtype=np.float32),
        "lugal": np.ones(768, dtype=np.float32) * 0.5,
    }
    morpheme_vectors["nar"][0] = 1.0  # king
    morpheme_vectors["ta"][1] = 1.0   # god

    misses = [
        {"sumerian": "nar-ta", "english": "king god", "confidence": 0.9},  # both morphemes in vocab
        {"sumerian": "nar-missing", "english": "x", "confidence": 0.9},    # "missing" not in vocab
        {"sumerian": "flat", "english": "x", "confidence": 0.9},           # no hyphen, not a candidate
    ]

    result = simulate_morpheme_composition(
        misses, ctx, morpheme_vector_lookup=morpheme_vectors.get
    )
    assert result["anchors_newly_resolvable_tier1"] == 1
    assert result["trustworthiness"].startswith("inferred")


def test_simulate_morpheme_composition_tier2_scores_against_expected_english():
    from scripts.coverage_diagnostic import simulate_morpheme_composition

    ctx = _tier2_ctx_with_exact_identity()
    # "nar-ta" morpheme-mean vector lands on index 0 (king) and index 1 (god) both at 0.5.
    morpheme_vectors = {
        "nar": np.zeros(768, dtype=np.float32),
        "ta": np.zeros(768, dtype=np.float32),
    }
    morpheme_vectors["nar"][0] = 1.0  # king
    morpheme_vectors["ta"][0] = 1.0   # reinforces king
    misses = [
        {"sumerian": "nar-ta", "english": "king", "confidence": 0.9},  # expected to be top-1
    ]
    result = simulate_morpheme_composition(
        misses, ctx, morpheme_vector_lookup=morpheme_vectors.get
    )
    tier2 = result["tier2_semantic"]
    assert tier2["tested"] == 1
    assert tier2["top1_correct"] == 1


def test_simulate_subword_inference_tier1_counts_above_threshold_overlap():
    from scripts.coverage_diagnostic import simulate_subword_inference, _ngrams

    ctx = _tier2_ctx_with_exact_identity()
    trained = _ngrams("narta", 3, 6)
    misses = [
        {"sumerian": "narta", "english": "king", "confidence": 0.9},       # 100% overlap
        {"sumerian": "unrelated", "english": "god", "confidence": 0.9},    # 0% overlap
    ]
    # Subword simulator uses a stub FastText that returns index-0 basis vector.
    def fake_subword_vector(word: str) -> np.ndarray:
        v = np.zeros(768, dtype=np.float32)
        v[0] = 1.0  # always 'king'
        return v

    result = simulate_subword_inference(
        misses, ctx, trained_ngrams=trained,
        subword_vector_lookup=fake_subword_vector,
        fasttext_min_n=3, fasttext_max_n=6,
    )
    assert result["anchors_newly_resolvable_tier1"] == 1
    tier2 = result["tier2_semantic"]
    assert tier2["tested"] == 1
    assert tier2["top1_correct"] == 1


def test_simulate_subword_inference_tier2_skips_anchors_without_english_in_gemma():
    from scripts.coverage_diagnostic import simulate_subword_inference, _ngrams

    ctx = _tier2_ctx_with_exact_identity()
    trained = _ngrams("narta", 3, 6)
    # Anchor's English side is "unlisted" which is NOT in ctx.gemma_vocab.
    misses = [
        {"sumerian": "narta", "english": "unlisted", "confidence": 0.9},
    ]
    def fake_subword_vector(word):
        v = np.zeros(768, dtype=np.float32)
        v[0] = 1.0
        return v

    result = simulate_subword_inference(
        misses, ctx, trained_ngrams=trained,
        subword_vector_lookup=fake_subword_vector,
        fasttext_min_n=3, fasttext_max_n=6,
    )
    # Tier-1 recoverable (ngram overlap passes).
    assert result["anchors_newly_resolvable_tier1"] == 1
    # Tier-2 skipped because expected english isn't in Gemma vocab.
    tier2 = result["tier2_semantic"]
    assert tier2["tested"] == 0
    assert tier2["skipped"] == 1
```

- [ ] **Step 2: Run new tests, verify they fail**

```bash
pytest tests/test_coverage_diagnostic.py -v
```
Expected: 7 new inference/Tier-2 tests FAIL on `ImportError`. Prior 22 still pass.

- [ ] **Step 3: Append inference simulators + Tier-2 helpers**

Append:

```python

# --- Tier-2 semantic validation --------------------------------------------


def _l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def _l2_normalize_vec(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm == 0:
        return v
    return v / norm


def _project_ft_to_gemma(ft_vec: np.ndarray, ctx: "DiagnosticContext") -> np.ndarray:
    """Fuse with zero-padding, project through Gemma ridge."""
    ft_vec = ft_vec.astype(np.float32)
    fused = np.concatenate([ft_vec, np.zeros(768, dtype=np.float32)])  # (1536,)
    projected = fused @ ctx.ridge_gemma_coef.T + ctx.ridge_gemma_intercept  # (768,)
    return projected


def _tier2_nearest_english(ft_vec: np.ndarray, ctx: "DiagnosticContext", k: int) -> list[str]:
    projected = _project_ft_to_gemma(ft_vec, ctx)
    query = _l2_normalize_vec(projected)
    eng_norm = _l2_normalize_rows(ctx.gemma_english_vectors)
    sims = eng_norm @ query  # (N,)
    top_idx = np.argsort(sims)[::-1][:k]
    return [ctx.gemma_english_vocab[int(i)] for i in top_idx]


def _tier2_score_anchor(
    ft_vec: np.ndarray,
    expected_english: str,
    ctx: "DiagnosticContext",
) -> dict:
    top10 = _tier2_nearest_english(ft_vec, ctx, k=10)
    expected = expected_english.lower()
    return {
        "top1": len(top10) >= 1 and top10[0] == expected,
        "top5": expected in top10[:5],
        "top10": expected in top10[:10],
    }


# --- Inference simulators --------------------------------------------------


def simulate_morpheme_composition(
    misses: list[dict],
    ctx: "DiagnosticContext",
    *,
    morpheme_vector_lookup,
) -> dict:
    """Simulator #4: hyphenated anchors; all morphemes in vocab; vector = mean."""
    resolvable_tier1 = 0
    tier2_tested = 0
    tier2_top1 = 0
    tier2_top5 = 0
    tier2_top10 = 0
    tier2_skipped = 0

    for anchor in misses:
        sumerian_raw = str(anchor.get("sumerian") or "").strip()
        morphemes = _morphemes(sumerian_raw)
        if not morphemes:
            continue
        if not all(m in ctx.fused_vocab for m in morphemes):
            continue

        resolvable_tier1 += 1

        # Tier-2: check if expected English is in Gemma vocab.
        expected = str(anchor.get("english") or "").lower()
        if expected not in ctx.gemma_vocab:
            tier2_skipped += 1
            continue

        # Synthesize morpheme-mean vector.
        vecs = []
        for m in morphemes:
            v = morpheme_vector_lookup(m)
            if v is None:
                break
            vecs.append(np.asarray(v, dtype=np.float32))
        if len(vecs) != len(morphemes):
            # Lookup missed a morpheme (shouldn't happen per vocab check, but guard).
            tier2_skipped += 1
            continue
        synthesized = np.mean(np.stack(vecs, axis=0), axis=0)

        score = _tier2_score_anchor(synthesized, expected, ctx)
        tier2_tested += 1
        if score["top1"]: tier2_top1 += 1
        if score["top5"]: tier2_top5 += 1
        if score["top10"]: tier2_top10 += 1

    return {
        "anchors_newly_resolvable_tier1": resolvable_tier1,
        "tier2_semantic": {
            "tested": tier2_tested,
            "top1_correct": tier2_top1,
            "top5_correct": tier2_top5,
            "top10_correct": tier2_top10,
            "skipped": tier2_skipped,
        },
        "trustworthiness": "inferred (compositional)",
        "notes": "Vector = numpy mean of constituent morpheme vectors. Tier-2 checks whitened-Gemma projection.",
    }


def simulate_subword_inference(
    misses: list[dict],
    ctx: "DiagnosticContext",
    *,
    trained_ngrams: frozenset[str],
    subword_vector_lookup,
    fasttext_min_n: int = 3,
    fasttext_max_n: int = 6,
) -> dict:
    """Simulator #5: FastText OOV inference with >= SUBWORD_OVERLAP_MIN n-gram overlap."""
    resolvable_tier1 = 0
    tier2_tested = 0
    tier2_top1 = 0
    tier2_top5 = 0
    tier2_top10 = 0
    tier2_skipped = 0

    for anchor in misses:
        sumerian_raw = str(anchor.get("sumerian") or "").strip()
        normalized = normalize_anchor_form(sumerian_raw)
        if not normalized:
            continue
        # Skip anchors already in vocab (not misses) — defensive.
        if normalized in ctx.fused_vocab:
            continue
        overlap = _subword_overlap(normalized, trained_ngrams, fasttext_min_n, fasttext_max_n)
        if overlap < SUBWORD_OVERLAP_MIN:
            continue

        resolvable_tier1 += 1

        expected = str(anchor.get("english") or "").lower()
        if expected not in ctx.gemma_vocab:
            tier2_skipped += 1
            continue

        ft_vec = subword_vector_lookup(normalized)
        if ft_vec is None:
            tier2_skipped += 1
            continue
        ft_vec = np.asarray(ft_vec, dtype=np.float32)

        score = _tier2_score_anchor(ft_vec, expected, ctx)
        tier2_tested += 1
        if score["top1"]: tier2_top1 += 1
        if score["top5"]: tier2_top5 += 1
        if score["top10"]: tier2_top10 += 1

    return {
        "anchors_newly_resolvable_tier1": resolvable_tier1,
        "tier2_semantic": {
            "tested": tier2_tested,
            "top1_correct": tier2_top1,
            "top5_correct": tier2_top5,
            "top10_correct": tier2_top10,
            "skipped": tier2_skipped,
        },
        "trustworthiness": "inferred (character n-gram)",
        "notes": "Uses FastText.wv.get_vector for OOV. Tier-2 checks whitened-Gemma projection.",
    }
```

- [ ] **Step 4: Run tests, verify all pass**

```bash
pytest tests/test_coverage_diagnostic.py -v
```
Expected: all 29 tests PASS (22 prior + 7 new).

- [ ] **Step 5: Commit**

```bash
cd /Users/crashy/Development/cuneiformy
git add scripts/coverage_diagnostic.py tests/test_coverage_diagnostic.py
git commit -m "feat: add coverage diagnostic inference simulators with Tier-2 validation"
```

---

## Task 5: Report rendering + main() + baseline run + journal

**Files:**
- Modify: `scripts/coverage_diagnostic.py` (append rendering + main + CLI)
- Modify: `tests/test_coverage_diagnostic.py` (append rendering/main tests)
- Create: `results/coverage_diagnostic_2026-04-19.{md,json}` (via the actual run in Step 6)
- Modify: `docs/EXPERIMENT_JOURNAL.md`

### Setup note

This task wires up:
- Deterministic example sampling (same pattern as audit).
- `render_json` and `render_markdown` report functions.
- Ranked intervention table (sorts interventions by projected-survives descending).
- `run_diagnostic` programmatic entry + `main` CLI wrapper.
- The baseline run on real data and journal entry.

The `run_diagnostic` function loads artifacts, finds misses by delegating to the audit's `classify_all`, builds a `DiagnosticContext`, runs classifier + all 5 simulators, renders + writes both reports, prints summary, returns 0.

- [ ] **Step 1: Append rendering + main tests**

Append:

```python
# --- Rendering + main tests ------------------------------------------------


def _full_metadata():
    return {
        "diagnostic_date": "2026-04-19",
        "source_artifacts": {
            "anchors_path": "data/processed/english_anchors.json",
            "anchors_sha256": "0" * 64,
            "fasttext_model_path": "models/fasttext_sumerian.model",
            "fasttext_model_sha256": "0" * 64,
            "fused_vocab_path": "models/fused_embeddings_1536d.npz",
            "glove_path": "data/processed/glove.6B.300d.txt",
            "gemma_path": "models/english_gemma_whitened_768d.npz",
            "ridge_gemma_path": "models/ridge_weights_gemma_whitened.npz",
            "oracc_lemmas_path": "data/raw/oracc_lemmas.json",
            "cleaned_corpus_path": "data/processed/cleaned_corpus.txt",
            "cleaned_corpus_sha256": "0" * 64,
            "seed": 42,
            "subword_overlap_min": 0.5,
        },
        "baseline": {
            "total_merged": 13886,
            "survives": 1951,
            "sumerian_vocab_miss": 4,
        },
    }


def _classifier_result_for_render():
    return {
        "total_misses": 4,
        "primary_causes": {
            "normalization_recoverable":        {"count": 1, "pct": 25.0, "rows": [{"sumerian": "za3sze3la2", "english": "force", "confidence": 0.9, "trace": {"normalized_form": "za3sze3la2"}}]},
            "in_corpus_below_min_count":        {"count": 1, "pct": 25.0, "rows": [{"sumerian": "rare3", "english": "x", "confidence": 0.9, "trace": {"matched_form": "rare3", "corpus_count": 3}}]},
            "oracc_lemma_surface_recoverable":  {"count": 0, "pct": 0.0, "rows": []},
            "morpheme_composition_recoverable": {"count": 1, "pct": 25.0, "rows": [{"sumerian": "nar-ta", "english": "musician", "confidence": 0.9, "trace": {"morphemes_in_vocab": ["nar", "ta"]}}]},
            "subword_inference_recoverable":    {"count": 0, "pct": 0.0, "rows": []},
            "genuinely_missing":                {"count": 1, "pct": 25.0, "rows": [{"sumerian": "weird", "english": "absent", "confidence": 0.9, "trace": {}}]},
        },
    }


def _simulator_result_for_render():
    return {
        "interventions": {
            "ascii_normalize": {"anchors_newly_resolvable": 1, "trustworthiness": "exact", "notes": "..."},
            "lower_min_count": {"per_threshold": {"1": {"anchors_newly_resolvable": 3}, "2": {"anchors_newly_resolvable": 2}, "3": {"anchors_newly_resolvable": 1}, "4": {"anchors_newly_resolvable": 0}}, "trustworthiness": "exact", "notes": "..."},
            "oracc_lemma_expansion": {"anchors_newly_resolvable": 0, "surface_forms_added_to_vocab": 0, "trustworthiness": "exact", "notes": "..."},
            "morpheme_composition": {"anchors_newly_resolvable_tier1": 1, "tier2_semantic": {"tested": 1, "top1_correct": 1, "top5_correct": 1, "top10_correct": 1, "skipped": 0}, "trustworthiness": "inferred (compositional)", "notes": "..."},
            "subword_inference": {"anchors_newly_resolvable_tier1": 1, "tier2_semantic": {"tested": 1, "top1_correct": 0, "top5_correct": 1, "top10_correct": 1, "skipped": 0}, "trustworthiness": "inferred (character n-gram)", "notes": "..."},
        },
    }


def test_render_json_schema_version_and_structure():
    from scripts.coverage_diagnostic import render_json

    report = render_json(
        _classifier_result_for_render(),
        _simulator_result_for_render(),
        _full_metadata(),
        examples_per_bucket=1,
    )
    assert report["diagnostic_schema_version"] == 1
    assert report["diagnostic_date"] == "2026-04-19"
    assert "baseline" in report
    assert "classifier" in report
    assert "simulator" in report
    assert set(report["classifier"]["primary_causes"].keys()) == {
        "normalization_recoverable", "in_corpus_below_min_count",
        "oracc_lemma_surface_recoverable", "morpheme_composition_recoverable",
        "subword_inference_recoverable", "genuinely_missing",
    }
    for bucket in report["classifier"]["primary_causes"].values():
        assert "count" in bucket
        assert "examples" in bucket
        assert "rows" not in bucket  # heavy list stripped


def test_render_json_is_deterministic_with_seed():
    from scripts.coverage_diagnostic import render_json

    r1 = render_json(
        _classifier_result_for_render(),
        _simulator_result_for_render(),
        _full_metadata(),
        examples_per_bucket=10,
    )
    r2 = render_json(
        _classifier_result_for_render(),
        _simulator_result_for_render(),
        _full_metadata(),
        examples_per_bucket=10,
    )
    import json as _json
    assert _json.dumps(r1, sort_keys=True) == _json.dumps(r2, sort_keys=True)


def test_render_markdown_has_required_sections():
    from scripts.coverage_diagnostic import render_markdown

    md = render_markdown(
        _classifier_result_for_render(),
        _simulator_result_for_render(),
        _full_metadata(),
        examples_per_bucket=1,
    )
    assert "# Coverage Diagnostic" in md
    assert "## Baseline" in md
    assert "## Classifier" in md
    assert "## Simulator" in md
    assert "## Ranked intervention" in md
    assert "## Methodology notes" in md


def test_render_markdown_escapes_pipe_characters_in_cells():
    from scripts.coverage_diagnostic import render_markdown

    classifier_with_pipe = _classifier_result_for_render()
    classifier_with_pipe["primary_causes"]["genuinely_missing"]["rows"] = [
        {"sumerian": "|AN.USAN|", "english": "night", "confidence": 0.9, "trace": {}}
    ]
    classifier_with_pipe["primary_causes"]["genuinely_missing"]["count"] = 1

    md = render_markdown(
        classifier_with_pipe,
        _simulator_result_for_render(),
        _full_metadata(),
        examples_per_bucket=1,
    )
    assert r"\|AN.USAN\|" in md


def test_run_diagnostic_exits_zero_with_synthetic_inputs(tmp_path, monkeypatch):
    """End-to-end: run_diagnostic produces both report files with synthetic artifacts."""
    import scripts.coverage_diagnostic as mod

    # Build synthetic artifacts in tmp_path.
    anchors = [
        {"sumerian": "lugal", "english": "king", "confidence": 0.9, "source": "ePSD2"},      # survives
        {"sumerian": "completelyunknown", "english": "king", "confidence": 0.9, "source": "ePSD2"},  # miss
    ]
    anchors_path = tmp_path / "anchors.json"
    anchors_path.write_text(json.dumps(anchors), encoding="utf-8")

    fused_path = tmp_path / "fused.npz"
    np.savez_compressed(
        str(fused_path),
        vectors=np.zeros((1, 1536), dtype=np.float32),
        vocab=np.array(["lugal"]),
    )

    glove_path = tmp_path / "glove.txt"
    glove_path.write_text("king " + " ".join(["0.0"] * 300) + "\n", encoding="utf-8")

    gemma_path = tmp_path / "gemma.npz"
    np.savez_compressed(
        str(gemma_path),
        vectors=np.eye(1, 768, dtype=np.float32),
        vocab=np.array(["king"]),
    )

    ridge_path = tmp_path / "ridge.npz"
    coef = np.zeros((768, 1536), dtype=np.float32)
    for i in range(768):
        coef[i, i] = 1.0
    np.savez_compressed(str(ridge_path), coef=coef, intercept=np.zeros(768, dtype=np.float32))

    lemmas_path = tmp_path / "lemmas.json"
    lemmas_path.write_text(json.dumps([]), encoding="utf-8")

    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text("lugal lugal\n", encoding="utf-8")

    # Train a tiny real FastText so subword inference path works.
    from gensim.models import FastText
    tiny_corpus = [["lugal", "dingir", "nar", "ta", "mu"]] * 5
    tiny_model = FastText(vector_size=768, min_count=1, sg=1, workers=1, epochs=1)
    tiny_model.build_vocab(corpus_iterable=tiny_corpus)
    tiny_model.train(corpus_iterable=tiny_corpus, total_examples=5, epochs=1)
    model_path = tmp_path / "fasttext.model"
    tiny_model.save(str(model_path))

    out_dir = tmp_path / "results"

    exit_code = mod.run_diagnostic(
        anchors_path=anchors_path,
        fused_path=fused_path,
        glove_path=glove_path,
        gemma_path=gemma_path,
        ridge_gemma_path=ridge_path,
        oracc_lemmas_path=lemmas_path,
        cleaned_corpus_path=corpus_path,
        fasttext_model_path=model_path,
        out_dir=out_dir,
        diagnostic_date="2026-04-19",
    )
    assert exit_code == 0
    assert (out_dir / "coverage_diagnostic_2026-04-19.md").exists()
    assert (out_dir / "coverage_diagnostic_2026-04-19.json").exists()

    report = json.loads((out_dir / "coverage_diagnostic_2026-04-19.json").read_text())
    assert report["diagnostic_schema_version"] == 1
    assert report["classifier"]["total_misses"] == 1


def test_run_diagnostic_raises_on_missing_required_input(tmp_path):
    import scripts.coverage_diagnostic as mod

    with pytest.raises(FileNotFoundError):
        mod.run_diagnostic(
            anchors_path=tmp_path / "missing.json",
            fused_path=tmp_path / "missing.npz",
            glove_path=tmp_path / "missing.txt",
            gemma_path=tmp_path / "missing.npz",
            ridge_gemma_path=tmp_path / "missing.npz",
            oracc_lemmas_path=tmp_path / "missing.json",
            cleaned_corpus_path=tmp_path / "missing.txt",
            fasttext_model_path=tmp_path / "missing.model",
            out_dir=tmp_path,
            diagnostic_date="2026-04-19",
        )
```

- [ ] **Step 2: Run new tests, verify they fail**

```bash
pytest tests/test_coverage_diagnostic.py -v
```
Expected: 7 new rendering/main tests FAIL on `ImportError`. Prior 29 still pass.

- [ ] **Step 3: Append rendering + main code**

Append:

```python

# --- Rendering -------------------------------------------------------------


RECOVERABILITY_TAGS = {
    "normalization_recoverable":        "exact — normalize and rerun extraction",
    "in_corpus_below_min_count":        "exact — lower min_count and retrain",
    "oracc_lemma_surface_recoverable":  "exact — expand anchors to ORACC surface forms",
    "morpheme_composition_recoverable": "inferred — compose vector from morphemes in vocab",
    "subword_inference_recoverable":    "inferred — use FastText OOV subword inference",
    "genuinely_missing":                "none — requires new corpus or lemma data",
}


def _escape_md_cell(s) -> str:
    return str(s).replace("|", r"\|")


def _pick_examples(rows: list[dict], n: int, seed: int) -> list[dict]:
    if not rows:
        return []
    n = min(n, len(rows))
    rng = np.random.default_rng(seed)
    indices = sorted(rng.choice(len(rows), size=n, replace=False).tolist())
    return [rows[i] for i in indices]


def render_json(
    classifier_result: dict,
    simulator_result: dict,
    metadata: dict,
    examples_per_bucket: int = 10,
) -> dict:
    seed = metadata.get("source_artifacts", {}).get("seed", DEFAULT_SEED)
    primary_causes_out = {}
    for name in PRIMARY_CAUSE_ORDER:
        bucket = classifier_result["primary_causes"][name]
        primary_causes_out[name] = {
            "count": bucket["count"],
            "pct": round(bucket["pct"], 4),
            "examples": _pick_examples(bucket.get("rows", []), examples_per_bucket, seed),
        }
    return {
        "diagnostic_schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "diagnostic_date": metadata["diagnostic_date"],
        "source_artifacts": metadata["source_artifacts"],
        "baseline": metadata["baseline"],
        "classifier": {
            "total_misses": classifier_result["total_misses"],
            "primary_causes": primary_causes_out,
        },
        "simulator": simulator_result,
    }


def _format_anchor_row(row: dict) -> str:
    trace = row.get("trace") or {}
    trace_str = ", ".join(f"{k}={v!r}" for k, v in trace.items())
    return (
        f"| {_escape_md_cell(row.get('sumerian', ''))} "
        f"| {_escape_md_cell(row.get('english', ''))} "
        f"| {row.get('confidence', 0):.3f} "
        f"| {_escape_md_cell(trace_str)} |"
    )


def _ranked_interventions(simulator_result: dict) -> list[tuple[str, int, str]]:
    """Return [(name, projected_count, trustworthiness_tag), ...] sorted desc by count.

    For lower_min_count uses t=1 (most permissive); for inference-based uses Tier-2 top-5.
    """
    interventions = simulator_result["interventions"]
    rows = []
    rows.append(("ascii_normalize",
                 interventions["ascii_normalize"]["anchors_newly_resolvable"],
                 interventions["ascii_normalize"]["trustworthiness"]))
    rows.append(("lower_min_count (t=1)",
                 interventions["lower_min_count"]["per_threshold"]["1"]["anchors_newly_resolvable"],
                 interventions["lower_min_count"]["trustworthiness"]))
    rows.append(("oracc_lemma_expansion",
                 interventions["oracc_lemma_expansion"]["anchors_newly_resolvable"],
                 interventions["oracc_lemma_expansion"]["trustworthiness"]))
    rows.append(("morpheme_composition (Tier-2 top-5)",
                 interventions["morpheme_composition"]["tier2_semantic"]["top5_correct"],
                 interventions["morpheme_composition"]["trustworthiness"]))
    rows.append(("subword_inference (Tier-2 top-5)",
                 interventions["subword_inference"]["tier2_semantic"]["top5_correct"],
                 interventions["subword_inference"]["trustworthiness"]))
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


def render_markdown(
    classifier_result: dict,
    simulator_result: dict,
    metadata: dict,
    examples_per_bucket: int = 10,
) -> str:
    baseline = metadata["baseline"]
    seed = metadata.get("source_artifacts", {}).get("seed", DEFAULT_SEED)
    total_misses = classifier_result["total_misses"]

    lines: list[str] = []
    lines.append(f"# Coverage Diagnostic — {metadata['diagnostic_date']}")
    lines.append("")
    lines.append("Generated by `scripts/coverage_diagnostic.py`. See design spec at "
                 "`docs/superpowers/specs/2026-04-19-coverage-diagnostic-design.md`.")
    lines.append("")

    lines.append("## Baseline")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Total merged anchors | {baseline['total_merged']:,} |")
    lines.append(f"| Surviving (both target vocabs) | {baseline['survives']:,} |")
    lines.append(f"| sumerian_vocab_miss (this diagnostic's input) | {baseline['sumerian_vocab_miss']:,} |")
    lines.append("")

    lines.append("## Classifier — primary-cause attribution (mutually exclusive)")
    lines.append("")
    lines.append("| Primary cause | Count | % of misses | Trustworthiness |")
    lines.append("|---|---:|---:|---|")
    for name in PRIMARY_CAUSE_ORDER:
        bucket = classifier_result["primary_causes"][name]
        lines.append(f"| `{name}` | {bucket['count']:,} | {bucket['pct']:.2f}% | {RECOVERABILITY_TAGS[name]} |")
    lines.append("")

    lines.append("## Classifier — example rows per bucket")
    lines.append("")
    lines.append(f"Up to {examples_per_bucket} deterministically-sampled rows per non-empty bucket.")
    lines.append("")
    for name in PRIMARY_CAUSE_ORDER:
        bucket = classifier_result["primary_causes"][name]
        rows = bucket.get("rows", [])
        examples = _pick_examples(rows, examples_per_bucket, seed)
        if not examples:
            continue
        lines.append(f"### `{name}` ({bucket['count']:,} rows)")
        lines.append("")
        lines.append("| sumerian | english | confidence | trace |")
        lines.append("|---|---|---:|---|")
        for row in examples:
            lines.append(_format_anchor_row(row))
        lines.append("")

    lines.append("## Simulator — per-intervention projected recovery")
    lines.append("")
    sim = simulator_result["interventions"]
    lines.append("### `ascii_normalize`")
    lines.append(f"- anchors_newly_resolvable: **{sim['ascii_normalize']['anchors_newly_resolvable']:,}**")
    lines.append(f"- trustworthiness: {sim['ascii_normalize']['trustworthiness']}")
    lines.append("")
    lines.append("### `lower_min_count`")
    lines.append("")
    lines.append("| min_count | anchors_newly_resolvable |")
    lines.append("|---:|---:|")
    for t in ("1", "2", "3", "4"):
        lines.append(f"| {t} | {sim['lower_min_count']['per_threshold'][t]['anchors_newly_resolvable']:,} |")
    lines.append(f"- trustworthiness: {sim['lower_min_count']['trustworthiness']}")
    lines.append("")
    lines.append("### `oracc_lemma_expansion`")
    lines.append(f"- anchors_newly_resolvable: **{sim['oracc_lemma_expansion']['anchors_newly_resolvable']:,}**")
    lines.append(f"- surface_forms_added_to_vocab: {sim['oracc_lemma_expansion']['surface_forms_added_to_vocab']:,}")
    lines.append(f"- trustworthiness: {sim['oracc_lemma_expansion']['trustworthiness']}")
    lines.append("")
    for name in ("morpheme_composition", "subword_inference"):
        inf = sim[name]
        t2 = inf["tier2_semantic"]
        lines.append(f"### `{name}`")
        lines.append(f"- anchors_newly_resolvable_tier1: **{inf['anchors_newly_resolvable_tier1']:,}**")
        lines.append(f"- Tier-2: tested={t2['tested']:,}, top1_correct={t2['top1_correct']:,}, "
                     f"top5_correct={t2['top5_correct']:,}, top10_correct={t2['top10_correct']:,}, "
                     f"skipped={t2.get('skipped', 0):,}")
        lines.append(f"- trustworthiness: {inf['trustworthiness']}")
        lines.append("")

    lines.append("## Ranked intervention recommendations")
    lines.append("")
    lines.append("| Intervention | Projected recoverable | Trustworthiness |")
    lines.append("|---|---:|---|")
    for name, count, tag in _ranked_interventions(simulator_result):
        lines.append(f"| {name} | {count:,} | {tag} |")
    lines.append("")

    lines.append("## Methodology notes")
    lines.append("")
    lines.append("- **Exact** trustworthiness = the intervention produces a vocab hit (no inference). "
                 "**Inferred** = the intervention synthesizes a vector from subwords/morphemes.")
    lines.append("- Tier-2 semantic validation projects synthesized Sumerian vectors through "
                 "`models/ridge_weights_gemma_whitened.npz` and checks whether the expected "
                 "English gloss is among the top-K Gemma nearest neighbors.")
    lines.append("- Interventions in the simulator table are INDEPENDENT — their counts can overlap. "
                 "The classifier table gives mutually-exclusive primary-cause attribution.")
    lines.append("- All input artifacts are SHA-256 stamped in the JSON report for future diff-runs.")
    lines.append("")

    return "\n".join(lines)


# --- Main ------------------------------------------------------------------

import argparse  # noqa: E402
import datetime as _dt  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def run_diagnostic(
    *,
    anchors_path: Path,
    fused_path: Path,
    glove_path: Path,
    gemma_path: Path,
    ridge_gemma_path: Path,
    oracc_lemmas_path: Path,
    cleaned_corpus_path: Path,
    fasttext_model_path: Path,
    out_dir: Path,
    diagnostic_date: str,
    examples_per_bucket: int = 10,
    seed: int = DEFAULT_SEED,
) -> int:
    # Hard require all inputs.
    for p in (anchors_path, fused_path, glove_path, gemma_path, ridge_gemma_path,
              oracc_lemmas_path, cleaned_corpus_path, fasttext_model_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Required input missing: {p}. "
                "See docs/superpowers/specs/2026-04-19-coverage-diagnostic-design.md for generators."
            )

    # Load everything.
    anchors = _load_anchors(anchors_path)
    fused_vocab = frozenset(_load_fused_vocab(fused_path))
    glove_vocab = frozenset(_load_glove_vocab(glove_path))
    gemma_vocab = frozenset(_load_gemma_vocab(gemma_path))
    gemma_english_vocab, gemma_english_vectors = _load_gemma_english_npz(gemma_path)
    ridge_coef, ridge_intercept = _load_ridge_weights(ridge_gemma_path)
    corpus_freq = _load_corpus_frequency(cleaned_corpus_path)
    lemma_map = _load_lemma_surface_map(oracc_lemmas_path)
    ft_model = _load_fasttext_model(fasttext_model_path)

    # Use the audit's classifier to find misses against BOTH target vocabs.
    # The sumerian_vocab_miss bucket is our input set.
    from scripts.audit_anchors import AuditContext, classify_all
    audit_ctx = AuditContext(
        fused_vocab=fused_vocab,
        glove_vocab=glove_vocab,
        gemma_vocab=gemma_vocab,
        collision_keys=frozenset(),  # irrelevant here; we only need the vocab_miss bucket
    )
    audit_result = classify_all(anchors, audit_ctx)
    misses = audit_result["buckets"]["sumerian_vocab_miss"]["rows"]
    baseline = {
        "total_merged": audit_result["totals"]["merged"],
        "survives": audit_result["totals"]["survives"],
        "sumerian_vocab_miss": audit_result["buckets"]["sumerian_vocab_miss"]["count"],
    }

    # Build the DiagnosticContext.
    ctx = DiagnosticContext(
        fused_vocab=fused_vocab,
        glove_vocab=glove_vocab,
        gemma_vocab=gemma_vocab,
        corpus_frequency=corpus_freq,
        lemma_surface_map=lemma_map,
        fasttext_model=ft_model,
        gemma_english_vocab=gemma_english_vocab,
        gemma_english_vectors=gemma_english_vectors,
        ridge_gemma_coef=ridge_coef,
        ridge_gemma_intercept=ridge_intercept,
    )

    # Trained n-grams (computed once, used by classifier + subword simulator).
    min_n = int(ft_model.wv.min_n)
    max_n = int(ft_model.wv.max_n)
    trained_ngrams = _trained_ngrams(fused_vocab, min_n, max_n)

    # Classifier.
    classifier_result = classify_all_misses(misses, ctx, trained_ngrams, min_n, max_n)

    # Morpheme lookup: use explicit vocab vectors from the FastText model.
    def morpheme_lookup(morpheme: str):
        if morpheme in ft_model.wv:
            return ft_model.wv[morpheme]
        return None

    # Subword lookup: FastText OOV inference via wv.get_vector.
    def subword_lookup(word: str):
        try:
            return ft_model.wv.get_vector(word)
        except KeyError:
            return None

    # Simulators.
    sim_ascii = simulate_ascii_normalize(misses, ctx)
    sim_lower = simulate_lower_min_count(misses, ctx)
    sim_lemma = simulate_oracc_lemma_expansion(misses, ctx)
    sim_morph = simulate_morpheme_composition(misses, ctx, morpheme_vector_lookup=morpheme_lookup)
    sim_sub = simulate_subword_inference(
        misses, ctx,
        trained_ngrams=trained_ngrams,
        subword_vector_lookup=subword_lookup,
        fasttext_min_n=min_n, fasttext_max_n=max_n,
    )
    simulator_result = {
        "interventions": {
            "ascii_normalize": sim_ascii,
            "lower_min_count": sim_lower,
            "oracc_lemma_expansion": sim_lemma,
            "morpheme_composition": sim_morph,
            "subword_inference": sim_sub,
        },
    }

    metadata = {
        "diagnostic_date": diagnostic_date,
        "source_artifacts": {
            "anchors_path": str(anchors_path),
            "anchors_sha256": _sha256(anchors_path),
            "fasttext_model_path": str(fasttext_model_path),
            "fasttext_model_sha256": _sha256(fasttext_model_path),
            "fused_vocab_path": str(fused_path),
            "glove_path": str(glove_path),
            "gemma_path": str(gemma_path),
            "ridge_gemma_path": str(ridge_gemma_path),
            "oracc_lemmas_path": str(oracc_lemmas_path),
            "cleaned_corpus_path": str(cleaned_corpus_path),
            "cleaned_corpus_sha256": _sha256(cleaned_corpus_path),
            "seed": seed,
            "subword_overlap_min": SUBWORD_OVERLAP_MIN,
        },
        "baseline": baseline,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"coverage_diagnostic_{diagnostic_date}.json"
    md_path = out_dir / f"coverage_diagnostic_{diagnostic_date}.md"

    json_report = render_json(classifier_result, simulator_result, metadata, examples_per_bucket=examples_per_bucket)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2)
        f.write("\n")

    md_report = render_markdown(classifier_result, simulator_result, metadata, examples_per_bucket=examples_per_bucket)
    md_path.write_text(md_report, encoding="utf-8")

    # Summary print.
    print(f"total_misses: {classifier_result['total_misses']:,}")
    for name in PRIMARY_CAUSE_ORDER:
        b = classifier_result["primary_causes"][name]
        print(f"  {name:>36s}: {b['count']:>6,}  ({b['pct']:.2f}%)")
    print(f"Report: {md_path}")
    print(f"Report: {json_path}")
    return 0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    root = Path(__file__).parent.parent
    parser = argparse.ArgumentParser(description="Sumerian anchor coverage diagnostic")
    parser.add_argument("--anchors", default=str(root / "data" / "processed" / "english_anchors.json"))
    parser.add_argument("--fused",   default=str(root / "models" / "fused_embeddings_1536d.npz"))
    parser.add_argument("--glove",   default=str(root / "data" / "processed" / "glove.6B.300d.txt"))
    parser.add_argument("--gemma",   default=str(root / "models" / "english_gemma_whitened_768d.npz"))
    parser.add_argument("--ridge-gemma", default=str(root / "models" / "ridge_weights_gemma_whitened.npz"))
    parser.add_argument("--oracc-lemmas", default=str(root / "data" / "raw" / "oracc_lemmas.json"))
    parser.add_argument("--cleaned-corpus", default=str(root / "data" / "processed" / "cleaned_corpus.txt"))
    parser.add_argument("--fasttext-model", default=str(root / "models" / "fasttext_sumerian.model"))
    parser.add_argument("--out-dir", default=str(root / "results"))
    parser.add_argument("--date", default=_dt.date.today().isoformat(),
                        help="Diagnostic date (YYYY-MM-DD), used in output filenames")
    parser.add_argument("--examples-per-bucket", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    return run_diagnostic(
        anchors_path=Path(args.anchors),
        fused_path=Path(args.fused),
        glove_path=Path(args.glove),
        gemma_path=Path(args.gemma),
        ridge_gemma_path=Path(args.ridge_gemma),
        oracc_lemmas_path=Path(args.oracc_lemmas),
        cleaned_corpus_path=Path(args.cleaned_corpus),
        fasttext_model_path=Path(args.fasttext_model),
        out_dir=Path(args.out_dir),
        diagnostic_date=args.date,
        examples_per_bucket=args.examples_per_bucket,
    )


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run all tests, verify all pass**

```bash
pytest tests/test_coverage_diagnostic.py -v
```
Expected: all 36 tests PASS (29 prior + 7 new rendering/main).

- [ ] **Step 5: Run full test suite to check for regressions**

```bash
pytest tests/ --ignore=tests/test_integration.py -q
```
Expected: 111 passed (75 prior + 36 new), 0 failed.

- [ ] **Step 6: Commit the code**

```bash
cd /Users/crashy/Development/cuneiformy
git add scripts/coverage_diagnostic.py tests/test_coverage_diagnostic.py
git commit -m "feat: add coverage diagnostic rendering, main, and full integration"
```

- [ ] **Step 7: Run the diagnostic on real data**

```bash
cd /Users/crashy/Development/cuneiformy
python scripts/coverage_diagnostic.py --date 2026-04-19
```

Capture the full stdout — you'll need the real numbers for the journal in Step 10.

If the run fails, read the error, fix the root cause (not the symptom), and re-run.

- [ ] **Step 8: Sanity-check the report pair**

```bash
python3 -c "
import json
from pathlib import Path
r = json.loads(Path('results/coverage_diagnostic_2026-04-19.json').read_text())
assert r['diagnostic_schema_version'] == 1
total = r['classifier']['total_misses']
bucket_sum = sum(b['count'] for b in r['classifier']['primary_causes'].values())
assert bucket_sum == total, f'bucket sum {bucket_sum} != total_misses {total}'
print('OK: classifier bucket sums match total_misses')
print(f'total_misses: {total:,}')
for name, bucket in r['classifier']['primary_causes'].items():
    print(f'  {name:>36s}: {bucket[\"count\"]:>6,}  ({bucket[\"pct\"]:.2f}%)')
print()
print('Simulator intervention recovery:')
sim = r['simulator']['interventions']
print(f'  ascii_normalize:      {sim[\"ascii_normalize\"][\"anchors_newly_resolvable\"]:>6,}')
for t in (\"1\", \"2\", \"3\", \"4\"):
    print(f'  lower_min_count(t={t}): {sim[\"lower_min_count\"][\"per_threshold\"][t][\"anchors_newly_resolvable\"]:>6,}')
print(f'  oracc_lemma_expansion: {sim[\"oracc_lemma_expansion\"][\"anchors_newly_resolvable\"]:>6,}  (+{sim[\"oracc_lemma_expansion\"][\"surface_forms_added_to_vocab\"]} surface forms)')
for name in (\"morpheme_composition\", \"subword_inference\"):
    i = sim[name]
    print(f'  {name:>24s}: tier1={i[\"anchors_newly_resolvable_tier1\"]:>6,}, tier2_top5={i[\"tier2_semantic\"][\"top5_correct\"]:>6,} (tested={i[\"tier2_semantic\"][\"tested\"]})')
"
```

Capture the full output. Two consecutive runs must produce byte-identical JSON — test that:

```bash
cp results/coverage_diagnostic_2026-04-19.json /tmp/cov_first.json
python scripts/coverage_diagnostic.py --date 2026-04-19
diff /tmp/cov_first.json results/coverage_diagnostic_2026-04-19.json
```
Expected: `diff` exits 0 with no output. If anything differs, determinism is broken — fix before committing.

- [ ] **Step 9: Commit the baseline reports**

```bash
cd /Users/crashy/Development/cuneiformy
git add -f results/coverage_diagnostic_2026-04-19.md results/coverage_diagnostic_2026-04-19.json
git commit -m "chore: commit 2026-04-19 coverage diagnostic baseline"
```

Note: `results/` is gitignored; `-f` force-adds the committed baseline, matching the Workstream 2a pattern.

- [ ] **Step 10: Add journal entry**

Read `docs/EXPERIMENT_JOURNAL.md` to find the insertion point: AFTER the preamble's closing `---` and BEFORE the 2026-04-18 Workstream 2a entry.

Insert this entry (use Bash heredoc if Edit is blocked by hooks). **Replace the `[PUT REAL NUMBERS HERE]` placeholders with values from your Step 8 output: real counts and percentages for the classifier buckets, and real per-intervention recovery numbers for the simulator.**

```markdown
## 2026-04-19 — Workstream 2b-pre: Coverage diagnostic baseline

**Hypothesis:** Workstream 2a showed `sumerian_vocab_miss: 11,798 (84.96%)` as the dominant dropout, but the initial "FastText retrain" framing was too narrow. An ML-engineer reassessment identified five candidate interventions (ASCII normalization, lower min_count, ORACC lemma surface expansion, morpheme composition, subword inference) that overlap in non-obvious ways. Before designing any fix, we need a data-driven attribution of which anchors are recoverable by which interventions.

**Method:** New standalone script `scripts/coverage_diagnostic.py` runs two independent analyses on the `sumerian_vocab_miss` set: a **classifier** (mutually-exclusive primary-cause attribution with 6 priority-ordered buckets) and a **simulator** (per-intervention projected recovery, with Tier-2 semantic validation for the two inference-based interventions via whitened-Gemma ridge projection). Reuses `scripts/audit_anchors.py` loaders and classifier to find misses. 36 unit tests against synthetic inputs and a toy FastText.

**Result — classifier (primary-cause attribution):** [PUT REAL NUMBERS HERE — e.g., "normalization_recoverable: N (X%)", "in_corpus_below_min_count: N (X%)", "oracc_lemma_surface_recoverable: N (X%)", "morpheme_composition_recoverable: N (X%)", "subword_inference_recoverable: N (X%)", "genuinely_missing: N (X%)"].

**Result — simulator (per-intervention recovery, independent):** [PUT REAL NUMBERS HERE — e.g., "ascii_normalize: N", "lower_min_count@t=1: N", "oracc_lemma_expansion: N (+M surface forms)", "morpheme_composition: tier1=N, tier2_top5=M", "subword_inference: tier1=N, tier2_top5=M"].

**Takeaway:** [PUT BRIEF INTERPRETATION HERE — e.g., "ORACC surface-form expansion is the top-yield intervention and requires no retrain. Morpheme composition Tier-2 top-5 recovers X anchors with semantically-correct projections. Combined with ORACC expansion and a modest min_count reduction, a portfolio of exact-match interventions can recover most of the 11,798 misses without any FastText retrain."].

**Artifacts:** `scripts/coverage_diagnostic.py`, `tests/test_coverage_diagnostic.py`, `results/coverage_diagnostic_2026-04-19.{md,json}`. Spec: `docs/superpowers/specs/2026-04-19-coverage-diagnostic-design.md`.
```

Replace ALL three `[PUT REAL NUMBERS HERE]` blocks with actual values before committing.

- [ ] **Step 11: Commit the journal entry**

```bash
cd /Users/crashy/Development/cuneiformy
git add docs/EXPERIMENT_JOURNAL.md
git commit -m "docs: journal Workstream 2b-pre coverage diagnostic baseline"
```

- [ ] **Step 12: Final full test run**

```bash
cd /Users/crashy/Development/cuneiformy
pytest tests/ --ignore=tests/test_integration.py -q
```
Expected: 111 passed, 0 failed.

---

## Self-Review

Spec requirements matched to tasks:

- **`scripts/coverage_diagnostic.py`** → Tasks 1 (loaders+context), 2 (classifier), 3 (exact simulators), 4 (inference simulators + Tier-2), 5 (rendering + main).
- **`tests/test_coverage_diagnostic.py`** → each task appends tests.
- **`results/coverage_diagnostic_2026-04-19.{md,json}`** → Task 5 Step 7 (generate), Step 9 (commit).
- **6-bucket priority-ordered classifier** → Task 2 Step 3 (code) + Task 2 Step 1 (tests).
- **5 interventions in simulator** → Tasks 3 + 4 cover all 5.
- **Tier-2 semantic validation** → Task 4 Step 3 (code).
- **Methodology constants (`SUBWORD_OVERLAP_MIN=0.5`, `DEFAULT_SEED=42`)** → Task 1 Step 3.
- **Dated markdown + JSON reports, byte-identical reruns, SHA-stamped inputs** → Task 5 Steps 3, 7, 8.
- **Fail fast on missing required inputs** → Task 5 `run_diagnostic` and test in Task 5 Step 1.
- **Journal entry** → Task 5 Step 10.
- **Byte-identical rerun check** → Task 5 Step 8.

Placeholder scan:
- Journal entry contains three explicit `[PUT REAL NUMBERS HERE]` blocks in Task 5 Step 10 with instructions to replace before committing. These are necessary because the real numbers are unknown until the diagnostic runs on real data.
- No `TBD`, `TODO`, "similar to", or "add appropriate error handling" patterns elsewhere.

Type consistency:
- `DiagnosticContext` fields consistent across Tasks 1 → 5.
- `classify_miss` signature (`anchor, ctx, trained_ngrams, fasttext_min_n=3, fasttext_max_n=6`) consistent across Task 2 Step 1 tests and Step 3 implementation.
- `simulate_*` function kwargs consistent across tests and implementations.
- `PRIMARY_CAUSE_ORDER` tuple defined once in Task 2; referenced by Tasks 2–5.
- `SUBWORD_OVERLAP_MIN = 0.5` defined in Task 1; used by Task 2 classifier and Task 4 simulator.
- `_pick_examples`, `_format_anchor_row`, `_escape_md_cell` — defined in Task 5 Step 3.
- `_ngrams`, `_trained_ngrams`, `_subword_overlap`, `normalize_anchor_form`, `_morphemes` — defined in Task 2 Step 3; consumed by Tasks 3 and 4.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-19-coverage-diagnostic.md`. Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task with two-stage review (spec compliance + code quality) per task. Matches Phase A, Phase B, and Workstream 2a.

**2. Inline Execution** — batch execution via `superpowers:executing-plans` with checkpoints.

Which approach?
