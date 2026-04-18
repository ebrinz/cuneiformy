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


# --- Rendering --------------------------------------------------------------

import numpy as np  # noqa: E402

RECOVERABILITY = {
    "junk_sumerian":       "low — upstream extraction bug",
    "duplicate_collision": "low — dedup by design",
    "low_confidence":      "medium — raise threshold only if false-positive rate checks out",
    "sumerian_vocab_miss": "high — candidate driver for Workstream 2b (FastText retrain)",
    "multiword_english":   "medium — cheap phrase-splitter or phrase-embedding could recover some",
    "english_both_miss":   "low — genuinely untranslatable or specialist vocab",
    "english_glove_miss":  "medium — check lemmatization / hyphenation variants",
    "english_gemma_miss":  "medium — same, Gemma-specific",
    "survives":            "n/a — already in the valid-anchor set",
}


def _pick_examples(rows: list[dict], n: int, seed: int) -> list[dict]:
    if not rows:
        return []
    n = min(n, len(rows))
    rng = np.random.default_rng(seed)
    indices = sorted(rng.choice(len(rows), size=n, replace=False).tolist())
    return [rows[i] for i in indices]


def render_json(result: dict, metadata: dict, examples_per_bucket: int = 10) -> dict:
    """Serializable report dict matching spec schema version 1."""
    seed = metadata.get("source_artifacts", {}).get("seed", DEFAULT_SEED)
    buckets_out = {}
    for name in BUCKET_ORDER:
        bucket = result["buckets"][name]
        buckets_out[name] = {
            "count": bucket["count"],
            "pct_total": round(bucket["pct_total"], 4),
            "examples": _pick_examples(bucket["rows"], examples_per_bucket, seed),
        }
    return {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "audit_date": metadata["audit_date"],
        "source_artifacts": metadata["source_artifacts"],
        "totals": result["totals"],
        "buckets": buckets_out,
        "english_venn": result["english_venn"],
    }


def _format_row(row: dict) -> str:
    return (
        f"| {row.get('sumerian','')} "
        f"| {row.get('english','')} "
        f"| {row.get('confidence', 0):.3f} "
        f"| {row.get('source','')} |"
    )


def render_markdown(result: dict, metadata: dict, examples_per_bucket: int = 10) -> str:
    totals = result["totals"]
    merged = totals["merged"]
    survives = totals["survives"]
    dropped = totals["dropped"]
    pct_survives = (survives / merged * 100.0) if merged else 0.0
    pct_dropped = (dropped / merged * 100.0) if merged else 0.0

    lines: list[str] = []
    lines.append(f"# Anchor Audit — {metadata['audit_date']}")
    lines.append("")
    lines.append("Generated by `scripts/audit_anchors.py`. See design spec at "
                 "`docs/superpowers/specs/2026-04-18-sumerian-anchor-audit-design.md`.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Count | % of total |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Total merged anchors | {merged:,} | 100.00% |")
    lines.append(f"| Surviving (both target vocabs) | {survives:,} | {pct_survives:.2f}% |")
    lines.append(f"| Dropped | {dropped:,} | {pct_dropped:.2f}% |")
    lines.append("")

    lines.append("## Dropout by bucket (priority-assigned, mutually exclusive)")
    lines.append("")
    lines.append("| Bucket | Count | % of total | % of dropped | Recoverability |")
    lines.append("|---|---:|---:|---:|---|")
    for name in BUCKET_ORDER:
        bucket = result["buckets"][name]
        count = bucket["count"]
        pct_total = bucket["pct_total"]
        pct_dropped_bucket = (count / dropped * 100.0) if dropped and name != "survives" else 0.0
        lines.append(
            f"| `{name}` | {count:,} | {pct_total:.2f}% | "
            f"{pct_dropped_bucket:.2f}% | {RECOVERABILITY[name]} |"
        )
    lines.append("")

    lines.append("## Cross-cut: English-side Venn")
    lines.append("")
    venn = result["english_venn"]
    lines.append("Over anchors that passed priority 1-5 checks and reached the English-side test.")
    lines.append("")
    lines.append("|  | In GloVe | Not in GloVe |")
    lines.append("|---|---:|---:|")
    lines.append(f"| **In Gemma** | {venn['in_glove_in_gemma']:,} | {venn['not_glove_in_gemma']:,} |")
    lines.append(f"| **Not in Gemma** | {venn['in_glove_not_gemma']:,} | {venn['not_glove_not_gemma']:,} |")
    lines.append("")

    lines.append("## Bucket examples")
    lines.append("")
    lines.append(f"Up to {examples_per_bucket} deterministically-sampled rows per non-empty bucket.")
    lines.append("")
    seed = metadata.get("source_artifacts", {}).get("seed", DEFAULT_SEED)
    for name in BUCKET_ORDER:
        bucket = result["buckets"][name]
        examples = _pick_examples(bucket["rows"], examples_per_bucket, seed)
        if not examples:
            continue
        lines.append(f"### `{name}` ({bucket['count']:,} rows)")
        lines.append("")
        lines.append("| sumerian | english | confidence | source |")
        lines.append("|---|---|---:|---|")
        for row in examples:
            lines.append(_format_row(row))
        lines.append("")

    lines.append("## Recoverability narrative")
    lines.append("")
    lines.append(_recoverability_narrative(result))
    lines.append("")

    return "\n".join(lines)


def _recoverability_narrative(result: dict) -> str:
    buckets = result["buckets"]
    dropped = result["totals"]["dropped"]
    if dropped == 0:
        return "All anchors survived — no dropouts to explain."

    high = buckets["sumerian_vocab_miss"]["count"]
    med = (
        buckets["low_confidence"]["count"]
        + buckets["multiword_english"]["count"]
        + buckets["english_glove_miss"]["count"]
        + buckets["english_gemma_miss"]["count"]
    )
    low = (
        buckets["junk_sumerian"]["count"]
        + buckets["duplicate_collision"]["count"]
        + buckets["english_both_miss"]["count"]
    )

    def pct(n: int) -> str:
        return f"{n / dropped * 100:.1f}%" if dropped else "0.0%"

    return (
        f"Of the {dropped:,} dropped anchors, "
        f"roughly {pct(high)} ({high:,}) are high-recoverability — Sumerian-side "
        "vocab gaps likely addressable by Workstream 2b (FastText retrain with "
        "different tokenization). "
        f"Another {pct(med)} ({med:,}) are medium-recoverability — "
        "phrase-splitting, lemmatization/hyphenation checks, or threshold "
        "tuning could recover some. "
        f"The remaining {pct(low)} ({low:,}) are low-recoverability — junk, "
        "by-design dedup losses, or specialist/untranslatable vocabulary."
    )
