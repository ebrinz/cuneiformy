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
