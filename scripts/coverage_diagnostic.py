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
