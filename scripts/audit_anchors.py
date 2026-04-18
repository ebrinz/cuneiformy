"""
Sumerian Anchor Quality Audit (Workstream 2a).

Diagnostic-only: classifies every merged anchor produced by
`scripts/06_extract_anchors.py` into mutually-exclusive survival/dropout
buckets against both the GloVe and whitened-Gemma English target vocabs.

Writes a dated report pair under `results/`:
  - anchor_audit_<YYYY-MM-DD>.md
  - anchor_audit_<YYYY-MM-DD>.json  (schema version 1)

See: docs/superpowers/specs/2026-04-18-sumerian-anchor-audit-design.md
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

# --- Bucket definitions -----------------------------------------------------

BUCKET_ORDER = (
    "junk_sumerian",
    "duplicate_collision",
    "low_confidence",
    "sumerian_vocab_miss",
    "multiword_english",
    "english_both_miss",
    "english_glove_miss",
    "english_gemma_miss",
    "survives",
)

_MULTIWORD_RE = re.compile(r"[\s_\-]")
AUDIT_SCHEMA_VERSION = 1
DEFAULT_SEED = 42


@dataclass(frozen=True)
class AuditContext:
    fused_vocab: frozenset[str]
    glove_vocab: frozenset[str]
    gemma_vocab: frozenset[str]
    collision_keys: frozenset[str]
    low_conf_threshold: float = 0.3

    def __post_init__(self):
        # Coerce any set-like input to frozenset for true immutability.
        for field_name in ("fused_vocab", "glove_vocab", "gemma_vocab", "collision_keys"):
            value = getattr(self, field_name)
            if not isinstance(value, frozenset):
                object.__setattr__(self, field_name, frozenset(value))


def _normalize_sumerian(raw) -> str:
    if raw is None:
        return ""
    return str(raw).strip()


def _is_junk_sumerian(s: str) -> bool:
    if not s:
        return True
    if len(s) <= 1:
        return True
    if not s.strip():
        return True
    return False


def _is_multiword(english_raw: str) -> bool:
    return bool(_MULTIWORD_RE.search(english_raw))


def classify_anchor(anchor: dict, ctx: AuditContext) -> str:
    """Return the bucket name for a single anchor, per priority ordering."""
    sumerian_raw = _normalize_sumerian(anchor.get("sumerian"))
    english_raw = str(anchor.get("english") or "").strip()
    confidence = float(anchor.get("confidence") or 0.0)

    if _is_junk_sumerian(sumerian_raw):
        return "junk_sumerian"
    if sumerian_raw in ctx.collision_keys:
        return "duplicate_collision"
    if confidence < ctx.low_conf_threshold:
        return "low_confidence"
    if sumerian_raw not in ctx.fused_vocab:
        return "sumerian_vocab_miss"
    if _is_multiword(english_raw):
        return "multiword_english"

    eng_lower = english_raw.lower()
    in_glove = eng_lower in ctx.glove_vocab
    in_gemma = eng_lower in ctx.gemma_vocab
    if not in_glove and not in_gemma:
        return "english_both_miss"
    if not in_glove:
        return "english_glove_miss"
    if not in_gemma:
        return "english_gemma_miss"
    return "survives"


def classify_all(anchors: Iterable[dict], ctx: AuditContext) -> dict:
    """Classify every anchor and return totals + per-bucket rows + Venn counts.

    The `english_venn` counts are computed ONLY over anchors that reached the
    English-side membership check — i.e., the union of the three `english_*_miss`
    buckets plus `survives`. Anchors that early-exited into junk, dedup-collision,
    low-confidence, sumerian_vocab_miss, or multiword_english buckets do NOT
    appear in the Venn. Therefore `sum(venn.values())` equals the size of those
    four buckets combined, not `totals.merged`.
    """
    anchors = list(anchors)
    buckets: dict[str, list[dict]] = {name: [] for name in BUCKET_ORDER}

    for anchor in anchors:
        bucket = classify_anchor(anchor, ctx)
        buckets[bucket].append(anchor)

    survives_count = len(buckets["survives"])
    merged = len(anchors)

    venn = {
        "in_glove_in_gemma": 0,
        "in_glove_not_gemma": 0,
        "not_glove_in_gemma": 0,
        "not_glove_not_gemma": 0,
    }
    for bucket_name in ("english_both_miss", "english_glove_miss", "english_gemma_miss", "survives"):
        for anchor in buckets[bucket_name]:
            eng_lower = str(anchor.get("english") or "").lower()
            in_glove = eng_lower in ctx.glove_vocab
            in_gemma = eng_lower in ctx.gemma_vocab
            if in_glove and in_gemma:
                venn["in_glove_in_gemma"] += 1
            elif in_glove and not in_gemma:
                venn["in_glove_not_gemma"] += 1
            elif not in_glove and in_gemma:
                venn["not_glove_in_gemma"] += 1
            else:
                venn["not_glove_not_gemma"] += 1

    return {
        "totals": {
            "merged": merged,
            "survives": survives_count,
            "dropped": merged - survives_count,
        },
        "buckets": {
            name: {
                "count": len(rows),
                "pct_total": (len(rows) / merged * 100.0) if merged else 0.0,
                "rows": rows,
            }
            for name, rows in buckets.items()
        },
        "english_venn": venn,
    }
