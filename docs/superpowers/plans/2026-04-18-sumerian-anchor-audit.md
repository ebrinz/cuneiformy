# Sumerian Anchor Audit (Workstream 2a) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `scripts/audit_anchors.py` plus tests — a diagnostic script that classifies every merged Sumerian↔English anchor into nine mutually-exclusive survival/dropout buckets and writes a dated markdown + JSON report against both the GloVe and whitened-Gemma target vocabs. Then run the audit and commit the baseline report pair for 2026-04-18.

**Architecture:** Pure-function core (classify + render) with a thin `main()` that handles I/O. No network, no modifications to any input, no coupling to the alignment pipeline. Reads four artifacts (anchors JSON, fused vocab npz, Gemma vocab npz, GloVe text file), replays the merge-dedup from `scripts/06_extract_anchors.py` when raw inputs are available (otherwise skips that bucket with a warning), emits two files under `results/`.

**Tech Stack:** Python 3, numpy, stdlib `json` / `hashlib` / `re` / `pathlib`, pytest. No new dependencies.

**Reference spec:** `docs/superpowers/specs/2026-04-18-sumerian-anchor-audit-design.md`

---

## Before You Begin

- Current branch: `master`. Before writing any code, cut a fresh feature branch:
  ```bash
  cd /Users/crashy/Development/cuneiformy
  git checkout -b feat/anchor-audit
  ```
  All task commits land on `feat/anchor-audit`. The branch-finishing step (merge to master) happens after Task 4 via `superpowers:finishing-a-development-branch`.

- Verify the input artifacts exist locally before starting Task 1 (Task 4 depends on all four; Tasks 1–3 can be developed against synthetic data):
  ```bash
  ls -la data/processed/english_anchors.json models/fused_embeddings_1536d.npz models/english_gemma_whitened_768d.npz data/processed/glove.6B.300d.txt
  ```
  If any are missing, note which and raise it before Task 4.

---

## File Structure

**New files:**
- `scripts/audit_anchors.py` — main script (~350 lines; pure-function core + thin `main`).
- `tests/test_audit_anchors.py` — unit tests (~300 lines; all synthetic inputs).
- `results/anchor_audit_2026-04-18.md` — generated baseline report (committed).
- `results/anchor_audit_2026-04-18.json` — generated baseline report (committed).

**Modified files:**
- `docs/EXPERIMENT_JOURNAL.md` — append Workstream 2a baseline entry.

**Consumed (existing artifacts, not modified):**
- `data/processed/english_anchors.json`
- `models/fused_embeddings_1536d.npz`
- `models/english_gemma_whitened_768d.npz`
- `data/processed/glove.6B.300d.txt`
- `data/raw/oracc_lemmas.json` — optional; used for `duplicate_collision` reconstruction.
- `data/raw/etcsl_texts.json` — optional; same.
- `scripts/06_extract_anchors.py` — `normalize_oracc_cf` helper is imported by the audit.

**Untouched:**
- All other pipeline scripts, `final_output/`, other tests.

---

## Task 1: Pure-function classification core (TDD)

**Files:**
- Create: `scripts/audit_anchors.py` (partial — pure-function core only this task)
- Create: `tests/test_audit_anchors.py`

### Setup note

This task builds the bucket classifier as a set of pure functions operating on in-memory dicts/sets. No file I/O, no real artifact dependency. The `AuditContext` dataclass holds the three vocab sets plus the collision-key set; tests build it in-memory.

- [ ] **Step 1: Ensure repo root is on `sys.path` for tests**

Check that `pytest.ini` contains `pythonpath = .` (it does, per Phase B). No change needed — just verify:
```bash
grep pythonpath pytest.ini
```

- [ ] **Step 2: Write the failing classifier tests**

Create `tests/test_audit_anchors.py` with:

```python
import json
import tempfile

import numpy as np
import pytest


# --- Classification tests ---------------------------------------------------


def _default_ctx(
    fused_vocab=None,
    glove_vocab=None,
    gemma_vocab=None,
    collision_keys=None,
    low_conf_threshold=0.3,
):
    from scripts.audit_anchors import AuditContext

    return AuditContext(
        fused_vocab=set(fused_vocab or {"lugal", "dingir"}),
        glove_vocab=set(glove_vocab or {"king", "god"}),
        gemma_vocab=set(gemma_vocab or {"king", "god"}),
        collision_keys=set(collision_keys or set()),
        low_conf_threshold=low_conf_threshold,
    )


def test_survives_requires_both_target_vocabs():
    from scripts.audit_anchors import classify_anchor

    ctx = _default_ctx(glove_vocab={"king"}, gemma_vocab={"king", "god"})
    bucket = classify_anchor(
        {"sumerian": "lugal", "english": "king", "confidence": 0.9, "source": "ePSD2"}, ctx
    )
    assert bucket == "survives"

    bucket2 = classify_anchor(
        {"sumerian": "dingir", "english": "god", "confidence": 0.9, "source": "ePSD2"}, ctx
    )
    assert bucket2 == "english_glove_miss"


def test_bucket_priority_ordering_sumerian_miss_beats_multiword():
    from scripts.audit_anchors import classify_anchor

    ctx = _default_ctx(fused_vocab={"dingir"})
    bucket = classify_anchor(
        {"sumerian": "e2.gal", "english": "royal palace", "confidence": 0.9, "source": "ePSD2"},
        ctx,
    )
    assert bucket == "sumerian_vocab_miss"


def test_junk_sumerian_detection():
    from scripts.audit_anchors import classify_anchor

    ctx = _default_ctx()
    for bad in ["", " ", "\t", "a", "\u200b"]:
        bucket = classify_anchor(
            {"sumerian": bad, "english": "king", "confidence": 0.9, "source": "ePSD2"}, ctx
        )
        assert bucket == "junk_sumerian", f"expected junk for {bad!r}, got {bucket}"


def test_low_confidence_bucket():
    from scripts.audit_anchors import classify_anchor

    ctx = _default_ctx(low_conf_threshold=0.3)
    bucket = classify_anchor(
        {"sumerian": "lugal", "english": "king", "confidence": 0.25, "source": "ETCSL"}, ctx
    )
    assert bucket == "low_confidence"


def test_duplicate_collision_bucket():
    from scripts.audit_anchors import classify_anchor

    ctx = _default_ctx(collision_keys={"lugal"})
    bucket = classify_anchor(
        {"sumerian": "lugal", "english": "monarch", "confidence": 0.5, "source": "ETCSL"}, ctx
    )
    assert bucket == "duplicate_collision"


def test_multiword_english_detection():
    from scripts.audit_anchors import classify_anchor

    ctx = _default_ctx()
    for phrase in ["royal palace", "to cut off", "sun-god", "cut_off"]:
        bucket = classify_anchor(
            {"sumerian": "lugal", "english": phrase, "confidence": 0.9, "source": "ePSD2"},
            ctx,
        )
        assert bucket == "multiword_english", f"expected multiword for {phrase!r}, got {bucket}"


def test_english_both_miss_when_neither_vocab_has_word():
    from scripts.audit_anchors import classify_anchor

    ctx = _default_ctx(glove_vocab={"god"}, gemma_vocab={"god"})
    bucket = classify_anchor(
        {"sumerian": "lugal", "english": "king", "confidence": 0.9, "source": "ePSD2"}, ctx
    )
    assert bucket == "english_both_miss"


def test_english_gemma_miss_when_only_glove_has_word():
    from scripts.audit_anchors import classify_anchor

    ctx = _default_ctx(glove_vocab={"king"}, gemma_vocab={"god"})
    bucket = classify_anchor(
        {"sumerian": "lugal", "english": "king", "confidence": 0.9, "source": "ePSD2"}, ctx
    )
    assert bucket == "english_gemma_miss"


def test_english_lookup_is_case_insensitive():
    from scripts.audit_anchors import classify_anchor

    ctx = _default_ctx(glove_vocab={"king"}, gemma_vocab={"king"})
    bucket = classify_anchor(
        {"sumerian": "lugal", "english": "KING", "confidence": 0.9, "source": "ePSD2"}, ctx
    )
    assert bucket == "survives"


def test_classify_all_sums_to_total():
    from scripts.audit_anchors import classify_all

    ctx = _default_ctx(fused_vocab={"lugal"}, glove_vocab={"king"}, gemma_vocab={"king"})
    anchors = [
        {"sumerian": "lugal", "english": "king", "confidence": 0.9, "source": "ePSD2"},
        {"sumerian": "missing", "english": "god", "confidence": 0.9, "source": "ePSD2"},
        {"sumerian": "", "english": "king", "confidence": 0.9, "source": "ePSD2"},
    ]
    result = classify_all(anchors, ctx)
    total = sum(b["count"] for b in result["buckets"].values())
    assert total == len(anchors) == 3
    assert result["buckets"]["survives"]["count"] == 1
    assert result["buckets"]["sumerian_vocab_miss"]["count"] == 1
    assert result["buckets"]["junk_sumerian"]["count"] == 1


def test_classify_all_empty_input():
    from scripts.audit_anchors import classify_all

    ctx = _default_ctx()
    result = classify_all([], ctx)
    assert result["totals"]["merged"] == 0
    assert result["totals"]["survives"] == 0
    assert result["totals"]["dropped"] == 0
    for bucket in result["buckets"].values():
        assert bucket["count"] == 0


def test_venn_accounting_over_single_token_anchors():
    from scripts.audit_anchors import classify_all

    ctx = _default_ctx(
        fused_vocab={"s1", "s2", "s3", "s4"},
        glove_vocab={"a", "b"},
        gemma_vocab={"a", "c"},
    )
    anchors = [
        {"sumerian": "s1", "english": "a", "confidence": 0.9, "source": "ePSD2"},
        {"sumerian": "s2", "english": "b", "confidence": 0.9, "source": "ePSD2"},
        {"sumerian": "s3", "english": "c", "confidence": 0.9, "source": "ePSD2"},
        {"sumerian": "s4", "english": "d", "confidence": 0.9, "source": "ePSD2"},
    ]
    result = classify_all(anchors, ctx)
    venn = result["english_venn"]
    assert venn["in_glove_in_gemma"] == 1
    assert venn["in_glove_not_gemma"] == 1
    assert venn["not_glove_in_gemma"] == 1
    assert venn["not_glove_not_gemma"] == 1
```

- [ ] **Step 3: Run tests, verify they fail**

```bash
cd /Users/crashy/Development/cuneiformy
pytest tests/test_audit_anchors.py -v
```
Expected: all 11 tests FAIL with `ModuleNotFoundError: No module named 'scripts.audit_anchors'` or `ImportError` on `AuditContext`.

- [ ] **Step 4: Implement the classification core**

Create `scripts/audit_anchors.py`:

```python
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
    fused_vocab: set[str]
    glove_vocab: set[str]
    gemma_vocab: set[str]
    collision_keys: set[str]
    low_conf_threshold: float = 0.3


def _normalize_sumerian(raw) -> str:
    """Lightweight normalization — anchors JSON already stores normalized form.

    Guards against None and strips whitespace.
    """
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
    english_raw = str(anchor.get("english") or "")
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
    """Classify every anchor and return totals + per-bucket rows + Venn counts."""
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
```

- [ ] **Step 5: Run tests, verify all pass**

```bash
pytest tests/test_audit_anchors.py -v
```
Expected: all 11 classification tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /Users/crashy/Development/cuneiformy
git add scripts/audit_anchors.py tests/test_audit_anchors.py
git commit -m "feat: add anchor audit classification core with 9-bucket taxonomy"
```

---

## Task 2: Report rendering + determinism (TDD)

**Files:**
- Modify: `scripts/audit_anchors.py` (append render functions)
- Modify: `tests/test_audit_anchors.py` (append render tests)

### Setup note

Task 1 produced `classify_all(...)` whose `buckets[name]["rows"]` lists are complete. This task adds:
- `_pick_examples(rows, n, seed)` — deterministic sampling.
- `render_json(result, metadata)` — JSON-serializable dict matching the spec schema.
- `render_markdown(result, metadata)` — human-readable report string.

The rows list stays in memory for example picking but does NOT get written to JSON — only the sampled examples do.

- [ ] **Step 1: Append render tests to `tests/test_audit_anchors.py`**

Append:

```python
# --- Render tests ----------------------------------------------------------


def _default_metadata():
    return {
        "audit_date": "2026-04-18",
        "source_artifacts": {
            "anchors_path": "data/processed/english_anchors.json",
            "anchors_sha256": "0" * 64,
            "fused_vocab_path": "models/fused_embeddings_1536d.npz",
            "fused_vocab_size": 2,
            "glove_path": "data/processed/glove.6B.300d.txt",
            "glove_vocab_size": 2,
            "gemma_path": "models/english_gemma_whitened_768d.npz",
            "gemma_vocab_size": 2,
            "seed": 42,
        },
    }


def test_render_json_schema_version_and_structure():
    from scripts.audit_anchors import classify_all, render_json

    ctx = _default_ctx()
    anchors = [{"sumerian": "lugal", "english": "king", "confidence": 0.9, "source": "ePSD2"}]
    result = classify_all(anchors, ctx)
    report = render_json(result, _default_metadata(), examples_per_bucket=1)

    assert report["audit_schema_version"] == 1
    assert report["audit_date"] == "2026-04-18"
    assert "source_artifacts" in report
    assert report["totals"]["merged"] == 1
    assert set(report["buckets"].keys()) == {
        "junk_sumerian", "duplicate_collision", "low_confidence",
        "sumerian_vocab_miss", "multiword_english", "english_both_miss",
        "english_glove_miss", "english_gemma_miss", "survives",
    }
    for bucket in report["buckets"].values():
        assert "count" in bucket
        assert "pct_total" in bucket
        assert "examples" in bucket
        assert isinstance(bucket["examples"], list)


def test_render_json_is_deterministic_with_seed():
    from scripts.audit_anchors import classify_all, render_json

    ctx = _default_ctx(fused_vocab={f"s{i}" for i in range(20)})
    anchors = [
        {"sumerian": f"s{i}", "english": f"e{i}", "confidence": 0.9, "source": "ePSD2"}
        for i in range(20)
    ]
    result = classify_all(anchors, ctx)
    report_a = render_json(result, _default_metadata(), examples_per_bucket=5)
    report_b = render_json(result, _default_metadata(), examples_per_bucket=5)

    import json as _json
    assert _json.dumps(report_a, sort_keys=True) == _json.dumps(report_b, sort_keys=True)


def test_render_json_examples_respect_per_bucket_cap():
    from scripts.audit_anchors import classify_all, render_json

    ctx = _default_ctx(fused_vocab={f"s{i}" for i in range(20)})
    anchors = [
        {"sumerian": f"s{i}", "english": "king", "confidence": 0.9, "source": "ePSD2"}
        for i in range(20)
    ]
    result = classify_all(anchors, ctx)
    report = render_json(result, _default_metadata(), examples_per_bucket=3)
    assert len(report["buckets"]["survives"]["examples"]) == 3


def test_render_markdown_contains_required_sections():
    from scripts.audit_anchors import classify_all, render_markdown

    ctx = _default_ctx()
    anchors = [{"sumerian": "lugal", "english": "king", "confidence": 0.9, "source": "ePSD2"}]
    result = classify_all(anchors, ctx)
    md = render_markdown(result, _default_metadata(), examples_per_bucket=1)

    assert "# Anchor Audit" in md
    assert "## Summary" in md
    assert "## Dropout by bucket" in md
    assert "## Cross-cut: English-side Venn" in md
    assert "## Bucket examples" in md
    assert "## Recoverability narrative" in md
    for bucket_name in ("junk_sumerian", "survives", "sumerian_vocab_miss"):
        assert bucket_name in md


def test_render_markdown_recoverability_heuristics_present():
    from scripts.audit_anchors import classify_all, render_markdown

    ctx = _default_ctx()
    anchors = [{"sumerian": "lugal", "english": "king", "confidence": 0.9, "source": "ePSD2"}]
    result = classify_all(anchors, ctx)
    md = render_markdown(result, _default_metadata(), examples_per_bucket=1)

    assert "high" in md.lower() or "medium" in md.lower() or "low" in md.lower()
```

- [ ] **Step 2: Run new tests, verify they fail**

```bash
pytest tests/test_audit_anchors.py -v
```
Expected: 5 new render tests FAIL with `ImportError: cannot import name 'render_json'` (or `render_markdown`). The 11 classification tests still PASS.

- [ ] **Step 3: Append render functions to `scripts/audit_anchors.py`**

Append:

```python

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
```

- [ ] **Step 4: Run all tests, verify all pass**

```bash
pytest tests/test_audit_anchors.py -v
```
Expected: all 16 tests PASS (11 classification + 5 render).

- [ ] **Step 5: Commit**

```bash
cd /Users/crashy/Development/cuneiformy
git add scripts/audit_anchors.py tests/test_audit_anchors.py
git commit -m "feat: add anchor audit report rendering (markdown + JSON)"
```

---

## Task 3: Artifact loaders + `main()` wiring

**Files:**
- Modify: `scripts/audit_anchors.py` (append loaders + main)
- Modify: `tests/test_audit_anchors.py` (append loader + integration tests)

### Setup note

This task wires pure-function classification + rendering to real I/O and adds a `_reconstruct_dedup_collisions` helper that gracefully skips when the raw-data files are absent. `main()` ties everything together via a CLI entry point.

- [ ] **Step 1: Append loader + main tests**

Append to `tests/test_audit_anchors.py`:

```python
# --- Loader and main integration tests -------------------------------------


def _write_anchors(tmp_path, rows):
    path = tmp_path / "english_anchors.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f)
    return path


def _write_fused_vocab(tmp_path, vocab):
    path = tmp_path / "fused_embeddings_1536d.npz"
    np.savez_compressed(
        str(path),
        vectors=np.zeros((len(vocab), 1536), dtype=np.float32),
        vocab=np.array(vocab),
    )
    return path


def _write_gemma_vocab(tmp_path, vocab):
    path = tmp_path / "english_gemma_whitened_768d.npz"
    np.savez_compressed(
        str(path),
        vectors=np.zeros((len(vocab), 768), dtype=np.float32),
        vocab=np.array(vocab),
    )
    return path


def _write_glove(tmp_path, vocab):
    path = tmp_path / "glove.6B.300d.txt"
    with open(path, "w", encoding="utf-8") as f:
        for word in vocab:
            zeros = " ".join(["0.0"] * 300)
            f.write(f"{word} {zeros}\n")
    return path


def test_load_fused_vocab_enforces_shape():
    from scripts.audit_anchors import _load_fused_vocab

    with tempfile.TemporaryDirectory() as tmp:
        import pathlib
        tmp_path = pathlib.Path(tmp)
        bad = tmp_path / "bad.npz"
        np.savez_compressed(
            str(bad),
            vectors=np.zeros((3, 999), dtype=np.float32),
            vocab=np.array(["a", "b", "c"]),
        )
        with pytest.raises(ValueError, match="1536"):
            _load_fused_vocab(bad)


def test_load_glove_vocab_reads_first_token_per_line():
    from scripts.audit_anchors import _load_glove_vocab

    with tempfile.TemporaryDirectory() as tmp:
        import pathlib
        tmp_path = pathlib.Path(tmp)
        path = _write_glove(tmp_path, ["king", "god", "palace"])
        vocab = _load_glove_vocab(path)
        assert vocab == {"king", "god", "palace"}


def test_load_glove_vocab_detects_malformed_row():
    from scripts.audit_anchors import _load_glove_vocab

    with tempfile.TemporaryDirectory() as tmp:
        import pathlib
        tmp_path = pathlib.Path(tmp)
        path = tmp_path / "bad.txt"
        with open(path, "w") as f:
            f.write("king " + " ".join(["0.0"] * 299) + "\n")
        with pytest.raises(ValueError, match="300"):
            _load_glove_vocab(path)


def test_main_writes_report_pair(tmp_path):
    import scripts.audit_anchors as mod

    anchors = [
        {"sumerian": "lugal", "english": "king", "confidence": 0.9, "source": "ePSD2"},
        {"sumerian": "", "english": "junk", "confidence": 0.9, "source": "ePSD2"},
    ]
    anchors_path = _write_anchors(tmp_path, anchors)
    fused_path = _write_fused_vocab(tmp_path, ["lugal"])
    gemma_path = _write_gemma_vocab(tmp_path, ["king"])
    glove_path = _write_glove(tmp_path, ["king"])

    out_dir = tmp_path / "results"
    out_dir.mkdir()

    exit_code = mod.run_audit(
        anchors_path=anchors_path,
        fused_path=fused_path,
        gemma_path=gemma_path,
        glove_path=glove_path,
        raw_oracc_path=None,
        raw_etcsl_path=None,
        out_dir=out_dir,
        audit_date="2026-04-18",
    )

    assert exit_code == 0
    md_path = out_dir / "anchor_audit_2026-04-18.md"
    json_path = out_dir / "anchor_audit_2026-04-18.json"
    assert md_path.exists()
    assert json_path.exists()

    report = json.loads(json_path.read_text())
    assert report["audit_schema_version"] == 1
    assert report["totals"]["merged"] == 2
    assert report["totals"]["survives"] == 1
    assert report["buckets"]["junk_sumerian"]["count"] == 1


def test_main_raises_on_missing_input(tmp_path):
    import scripts.audit_anchors as mod

    with pytest.raises(FileNotFoundError):
        mod.run_audit(
            anchors_path=tmp_path / "missing.json",
            fused_path=tmp_path / "missing.npz",
            gemma_path=tmp_path / "missing.npz",
            glove_path=tmp_path / "missing.txt",
            raw_oracc_path=None,
            raw_etcsl_path=None,
            out_dir=tmp_path,
            audit_date="2026-04-18",
        )


def test_vocab_lowercase_sample_check_raises_on_mixed_case(tmp_path):
    import scripts.audit_anchors as mod

    path = tmp_path / "bad_case.npz"
    np.savez_compressed(
        str(path),
        vectors=np.zeros((5, 768), dtype=np.float32),
        vocab=np.array(["KING", "god", "Palace", "water", "sky"]),
    )
    with pytest.raises(ValueError, match="lowercase"):
        mod._load_gemma_vocab(path)
```

- [ ] **Step 2: Run new tests, verify they fail**

```bash
pytest tests/test_audit_anchors.py -v
```
Expected: 6 new tests FAIL on `ImportError` or `AttributeError` — loaders and `run_audit` don't exist yet.

- [ ] **Step 3: Append loaders and main to `scripts/audit_anchors.py`**

Append:

```python

# --- Artifact loaders -------------------------------------------------------

import argparse  # noqa: E402
import datetime as _dt  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402


def _assert_lowercase_sample(vocab: list[str], source: str, sample_size: int = 100) -> None:
    sample = vocab[:sample_size]
    for word in sample:
        if word != word.lower():
            raise ValueError(
                f"{source} vocab is not lowercase — saw {word!r}. "
                "Audit expects pre-lowercased caches."
            )


def _load_fused_vocab(path: Path) -> set[str]:
    data = np.load(str(path), allow_pickle=True)
    vectors = data["vectors"]
    vocab = [str(w) for w in data["vocab"]]
    if vectors.shape[1] != 1536:
        raise ValueError(
            f"Fused vocab vectors dim {vectors.shape[1]} != 1536 — "
            "regenerate via scripts/08_fuse_embeddings.py"
        )
    if len(vocab) != vectors.shape[0]:
        raise ValueError("Fused vocab row/vector count mismatch")
    return set(vocab)


def _load_gemma_vocab(path: Path) -> set[str]:
    data = np.load(str(path))
    vectors = data["vectors"]
    vocab = [str(w) for w in data["vocab"]]
    if vectors.shape[1] != 768:
        raise ValueError(
            f"Gemma vocab vectors dim {vectors.shape[1]} != 768 — "
            "regenerate via scripts/whiten_gemma.py"
        )
    if len(vocab) != vectors.shape[0]:
        raise ValueError("Gemma vocab row/vector count mismatch")
    _assert_lowercase_sample(vocab, "Gemma")
    return set(vocab)


def _load_glove_vocab(path: Path) -> set[str]:
    vocab: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.rstrip("\n").split(" ")
            if len(parts) != 301:
                raise ValueError(
                    f"GloVe line {line_no} has {len(parts) - 1} dims, expected 300 — "
                    "verify data/processed/glove.6B.300d.txt integrity"
                )
            vocab.append(parts[0])
    _assert_lowercase_sample(vocab, "GloVe")
    return set(vocab)


def _load_anchors(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        anchors = json.load(f)
    bad_rows: list[tuple[int, dict]] = []
    for i, row in enumerate(anchors):
        if not isinstance(row, dict):
            bad_rows.append((i, row))
            continue
        if "sumerian" not in row or "english" not in row or "confidence" not in row:
            bad_rows.append((i, row))
        if len(bad_rows) >= 5:
            break
    if bad_rows:
        raise ValueError(
            f"Anchors JSON has malformed rows; first bad examples: {bad_rows}"
        )
    return anchors


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _reconstruct_dedup_collisions(
    raw_oracc_path: Path | None,
    raw_etcsl_path: Path | None,
    merged_anchors: list[dict],
) -> set[str]:
    """Return the set of Sumerian keys lost to merge_anchors' higher-confidence
    dedup. If raw inputs are missing, returns an empty set (and the
    duplicate_collision bucket will report 0)."""
    if raw_oracc_path is None or raw_etcsl_path is None:
        return set()
    if not raw_oracc_path.exists() or not raw_etcsl_path.exists():
        print(
            f"WARN: raw inputs not found at {raw_oracc_path} / {raw_etcsl_path}; "
            "duplicate_collision bucket will be 0.",
            file=sys.stderr,
        )
        return set()

    # The upstream file is named `06_extract_anchors.py`, which isn't directly
    # importable (leading digit). Use importlib.util to load it by path.
    import importlib.util
    root = Path(__file__).parent.parent
    spec_path = root / "scripts" / "06_extract_anchors.py"
    spec = importlib.util.spec_from_file_location("extract_anchors_06_mod", spec_path)
    extract_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extract_module)

    with open(raw_oracc_path) as f:
        lemmas = json.load(f)
    dict_anchors = extract_module.extract_epsd2_anchors(lemmas, min_occurrences=5)

    with open(raw_etcsl_path) as f:
        etcsl_lines = json.load(f)
    parallel = [line for line in etcsl_lines if line.get("translation")]
    cooc_anchors = extract_module.extract_cooccurrence_anchors(
        parallel, min_cooccurrences=3, min_confidence=0.3
    )

    pre_merge_keys = {a["sumerian"] for a in (dict_anchors + cooc_anchors)}
    post_merge_keys = {a["sumerian"] for a in merged_anchors}
    return pre_merge_keys - post_merge_keys


# --- Main -------------------------------------------------------------------


def run_audit(
    *,
    anchors_path: Path,
    fused_path: Path,
    gemma_path: Path,
    glove_path: Path,
    raw_oracc_path: Path | None,
    raw_etcsl_path: Path | None,
    out_dir: Path,
    audit_date: str,
    examples_per_bucket: int = 10,
    seed: int = DEFAULT_SEED,
) -> int:
    anchors = _load_anchors(anchors_path)
    fused_vocab = _load_fused_vocab(fused_path)
    gemma_vocab = _load_gemma_vocab(gemma_path)
    glove_vocab = _load_glove_vocab(glove_path)
    collision_keys = _reconstruct_dedup_collisions(raw_oracc_path, raw_etcsl_path, anchors)

    ctx = AuditContext(
        fused_vocab=fused_vocab,
        glove_vocab=glove_vocab,
        gemma_vocab=gemma_vocab,
        collision_keys=collision_keys,
    )

    result = classify_all(anchors, ctx)

    metadata = {
        "audit_date": audit_date,
        "source_artifacts": {
            "anchors_path": str(anchors_path),
            "anchors_sha256": _sha256(anchors_path),
            "fused_vocab_path": str(fused_path),
            "fused_vocab_size": len(fused_vocab),
            "glove_path": str(glove_path),
            "glove_vocab_size": len(glove_vocab),
            "gemma_path": str(gemma_path),
            "gemma_vocab_size": len(gemma_vocab),
            "seed": seed,
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"anchor_audit_{audit_date}.json"
    md_path = out_dir / f"anchor_audit_{audit_date}.md"

    json_report = render_json(result, metadata, examples_per_bucket=examples_per_bucket)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2)
        f.write("\n")

    md_report = render_markdown(result, metadata, examples_per_bucket=examples_per_bucket)
    md_path.write_text(md_report, encoding="utf-8")

    totals = result["totals"]
    pct = (totals["survives"] / totals["merged"] * 100.0) if totals["merged"] else 0.0
    print(
        f"survives: {totals['survives']:,}/{totals['merged']:,} ({pct:.2f}%), "
        f"dropped: {totals['dropped']:,}"
    )
    print(f"Report: {md_path}")
    print(f"Report: {json_path}")
    return 0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    root = Path(__file__).parent.parent
    parser = argparse.ArgumentParser(description="Sumerian anchor quality audit")
    parser.add_argument(
        "--anchors",
        default=str(root / "data" / "processed" / "english_anchors.json"),
    )
    parser.add_argument(
        "--fused",
        default=str(root / "models" / "fused_embeddings_1536d.npz"),
    )
    parser.add_argument(
        "--gemma",
        default=str(root / "models" / "english_gemma_whitened_768d.npz"),
    )
    parser.add_argument(
        "--glove",
        default=str(root / "data" / "processed" / "glove.6B.300d.txt"),
    )
    parser.add_argument(
        "--raw-oracc",
        default=str(root / "data" / "raw" / "oracc_lemmas.json"),
    )
    parser.add_argument(
        "--raw-etcsl",
        default=str(root / "data" / "raw" / "etcsl_texts.json"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(root / "results"),
    )
    parser.add_argument(
        "--date",
        default=_dt.date.today().isoformat(),
        help="Audit date (YYYY-MM-DD), used in output filenames",
    )
    parser.add_argument("--examples-per-bucket", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    raw_oracc_path = Path(args.raw_oracc) if args.raw_oracc else None
    raw_etcsl_path = Path(args.raw_etcsl) if args.raw_etcsl else None
    return run_audit(
        anchors_path=Path(args.anchors),
        fused_path=Path(args.fused),
        gemma_path=Path(args.gemma),
        glove_path=Path(args.glove),
        raw_oracc_path=raw_oracc_path,
        raw_etcsl_path=raw_etcsl_path,
        out_dir=Path(args.out_dir),
        audit_date=args.date,
        examples_per_bucket=args.examples_per_bucket,
    )


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run all tests, verify all pass**

```bash
pytest tests/test_audit_anchors.py -v
```
Expected: all 22 tests PASS (11 classification + 5 render + 6 loader/main).

- [ ] **Step 5: Run full test suite to check for regressions**

```bash
pytest tests/ --ignore=tests/test_integration.py -q
```
Expected: 72 passed (50 prior + 22 new), 0 failed.

- [ ] **Step 6: Commit**

```bash
cd /Users/crashy/Development/cuneiformy
git add scripts/audit_anchors.py tests/test_audit_anchors.py
git commit -m "feat: wire anchor audit loaders + run_audit entry point"
```

---

## Task 4: Run the audit on real data + commit baseline + journal

**Files:**
- Create: `results/anchor_audit_2026-04-18.md`
- Create: `results/anchor_audit_2026-04-18.json`
- Modify: `docs/EXPERIMENT_JOURNAL.md`

### Setup note

This is the delivery step. Task 3 tests already proved the pipeline works against synthetic inputs; this task runs it against the real repo artifacts and journals the baseline.

- [ ] **Step 1: Run the audit**

```bash
cd /Users/crashy/Development/cuneiformy
python scripts/audit_anchors.py --date 2026-04-18
```

Expected stdout (approximately; exact numbers will vary):
```
survives: 1,965/13,886 (14.15%), dropped: 11,921
Report: /Users/crashy/Development/cuneiformy/results/anchor_audit_2026-04-18.md
Report: /Users/crashy/Development/cuneiformy/results/anchor_audit_2026-04-18.json
```

If the run exits non-zero, read the error, fix the underlying issue, and re-run. Do not silently rerun in a loop.

- [ ] **Step 2: Sanity-check the report pair**

```bash
python3 -c "
import json
from pathlib import Path
r = json.loads(Path('results/anchor_audit_2026-04-18.json').read_text())
assert r['audit_schema_version'] == 1
totals = r['totals']
assert totals['merged'] == totals['survives'] + totals['dropped']
bucket_sum = sum(b['count'] for b in r['buckets'].values())
assert bucket_sum == totals['merged'], f'bucket sum {bucket_sum} != merged {totals[\"merged\"]}'
venn = r['english_venn']
venn_sum = sum(venn.values())
venn_denom = (
    r['buckets']['english_both_miss']['count']
    + r['buckets']['english_glove_miss']['count']
    + r['buckets']['english_gemma_miss']['count']
    + r['buckets']['survives']['count']
)
assert venn_sum == venn_denom, f'venn sum {venn_sum} != denom {venn_denom}'
print('OK: bucket sum, venn sum, schema version all consistent')
print(f'survives: {totals[\"survives\"]:,} / {totals[\"merged\"]:,}')
for name, bucket in r['buckets'].items():
    print(f'  {name:>22s}: {bucket[\"count\"]:>6,}  ({bucket[\"pct_total\"]:.2f}%)')
"
```

Expected: `OK: bucket sum, venn sum, schema version all consistent`, followed by the per-bucket breakdown. If any assertion trips, that's an internal-consistency bug — investigate before proceeding.

- [ ] **Step 3: Determinism check — run the audit a second time and diff**

```bash
cd /Users/crashy/Development/cuneiformy
cp results/anchor_audit_2026-04-18.json /tmp/anchor_audit_first.json
python scripts/audit_anchors.py --date 2026-04-18
diff /tmp/anchor_audit_first.json results/anchor_audit_2026-04-18.json
```
Expected: `diff` exits 0 with no output — the two runs produced byte-identical JSON. If `diff` finds differences, there's non-determinism somewhere; fix before committing.

- [ ] **Step 4: Commit the baseline reports**

```bash
cd /Users/crashy/Development/cuneiformy
git add results/anchor_audit_2026-04-18.md results/anchor_audit_2026-04-18.json
git commit -m "chore: commit 2026-04-18 anchor audit baseline"
```

- [ ] **Step 5: Add journal entry**

Append to `docs/EXPERIMENT_JOURNAL.md` as a new topmost dated entry (after the preamble, before the existing 2026-04-16 Phase B entry):

```markdown
## 2026-04-18 — Workstream 2a: Anchor audit baseline

**Hypothesis:** The 14.2% valid-anchor survival rate (1,965 / 13,886) is the cheapest place to recover alignment top-1. Before building fixes, we need a reproducible categorization of what happens to the other 85% against both target spaces (GloVe + whitened Gemma), not a hunch-driven one.

**Method:** New standalone script `scripts/audit_anchors.py` classifies every merged anchor into 9 mutually-exclusive, priority-assigned buckets: `junk_sumerian`, `duplicate_collision`, `low_confidence`, `sumerian_vocab_miss`, `multiword_english`, `english_both_miss`, `english_glove_miss`, `english_gemma_miss`, `survives`. Emits dated markdown + JSON reports (schema v1). Pure-function core, fully unit-tested against synthetic data (22 tests). Runs against committed input artifacts; reconstructs dedup collisions when raw extraction inputs are locally present.

**Result:** Baseline report committed at `results/anchor_audit_2026-04-18.{md,json}`. [Fill in the top-line numbers from the real run here: e.g., `sumerian_vocab_miss: N (X%)`, `multiword_english: N (X%)`, and total survives.] Bucket sums and English-side Venn cross-check reconcile; two consecutive runs are byte-identical.

**Takeaway:** Workstream 2a's methodology gate is closed. Any future pipeline change (new tokenization, new target space, new extraction rules) can now be scored by how it moves the bucket distribution, not by anecdote. The bucket-count deltas prioritize Workstream 2b (FastText retrain for `sumerian_vocab_miss` recovery), 2c (Gemma fine-tune for `english_gemma_miss` recovery), and any `multiword_english` phrase-handling work.

**Artifacts:** `scripts/audit_anchors.py`, `tests/test_audit_anchors.py`, `results/anchor_audit_2026-04-18.{md,json}`. Spec: `docs/superpowers/specs/2026-04-18-sumerian-anchor-audit-design.md`.
```

**Important:** The `[Fill in…]` placeholder must be replaced with the real top-line bucket percentages before committing. Read the markdown report, copy the three largest dropout buckets and their percentages into the paragraph.

- [ ] **Step 6: Commit the journal entry**

```bash
cd /Users/crashy/Development/cuneiformy
git add docs/EXPERIMENT_JOURNAL.md
git commit -m "docs: journal Workstream 2a anchor audit baseline"
```

- [ ] **Step 7: Run the full test suite one last time**

```bash
cd /Users/crashy/Development/cuneiformy
pytest tests/ --ignore=tests/test_integration.py -q
```
Expected: all 72 tests pass (50 prior + 22 new). No regressions.

---

## Self-Review

Spec requirements matched to tasks:

- **In-scope: `scripts/audit_anchors.py`** → Tasks 1 (classifier core), 2 (rendering), 3 (loaders + main).
- **In-scope: `tests/test_audit_anchors.py`** → Tasks 1, 2, 3 all append to it.
- **In-scope: `results/anchor_audit_2026-04-18.{md,json}`** → Task 4 Step 1 (generate) and Step 4 (commit).
- **In-scope: dual-target diagnosis (GloVe + Gemma)** → Task 1 `AuditContext` holds both vocabs; Task 3 loads both.
- **Out-of-scope items** (phrase-splitter, FastText retrain, threshold changes) → none appear as tasks.
- **Success criterion: script exits 0** → Task 4 Step 1.
- **Success criterion: bucket counts sum to totals.merged** → Task 4 Step 2 assertion.
- **Success criterion: English Venn reconciles** → Task 4 Step 2 assertion.
- **Success criterion: byte-identical reruns** → Task 4 Step 3.
- **Success criterion: unit tests pass** → Task 1 Step 5, Task 2 Step 4, Task 3 Step 4.
- **Success criterion: totals.survives roughly matches `final_output/metadata.json` 1965 to within Gemma-intersection tolerance** → Task 4 Step 1 prints the real number; any discrepancy is surfaced in the journal entry.
- **Error handling: FileNotFoundError on missing inputs** → Task 3 Step 1 test + loader implementations.
- **Error handling: ValueError on shape / lowercase / malformed-anchor** → Task 3 Step 1 tests + loader validation.
- **Error handling: raw inputs absent → skip dedup-collision** → Task 3 `_reconstruct_dedup_collisions` handles this gracefully.

Placeholder scan:
- One deliberate placeholder in the journal entry paragraph (`[Fill in…]`) is explicitly called out in Task 4 Step 5 with instructions to replace it from real data before committing. This is the only placeholder in the plan and it is required because the actual numbers are unknown until the audit runs.
- No `TBD`, `TODO`, "similar to", or "add appropriate error handling" patterns.

Type consistency:
- `AuditContext` used with consistent fields across Tasks 1, 2, 3.
- `classify_all` return shape (`totals`, `buckets[name].{count,pct_total,rows}`, `english_venn`) matches what `render_json` / `render_markdown` consume.
- `run_audit` keyword args in Task 3's test match the signature in Task 3's implementation.
- `BUCKET_ORDER` defined once in Task 1; referenced by name in Tasks 2 and 3. Spelling consistent across all tasks: `junk_sumerian`, `duplicate_collision`, `low_confidence`, `sumerian_vocab_miss`, `multiword_english`, `english_both_miss`, `english_glove_miss`, `english_gemma_miss`, `survives`.
- `DEFAULT_SEED = 42` defined in Task 1, used in Task 2's `_pick_examples` and Task 3's `run_audit` kwarg default.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-18-sumerian-anchor-audit.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Matches how Phase A and Phase B shipped.

**2. Inline Execution** — Execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints.

Which approach?
