# Sumerian Anchor Quality Audit (Workstream 2a)

**Date:** 2026-04-18
**Status:** Approved (brainstorming), pending writing-plans
**Branch:** `master` (to be cut into a fresh feature branch at implementation time)
**Follows:** `docs/superpowers/specs/2026-04-16-phase-b-gemma-downstream-design.md` (Phase B)
**Journal:** `docs/EXPERIMENT_JOURNAL.md`

## Summary

Build a standalone diagnostic script, `scripts/audit_anchors.py`, that classifies every merged anchor pair produced by `scripts/06_extract_anchors.py` into mutually-exclusive dropout/survival buckets and emits a dated report (markdown + JSON). The report is the methodology — a reproducible, diffable snapshot of what happens to the 13,886 → 1,965 anchor pipeline under both the GloVe and whitened-Gemma target spaces. Fixes are out of scope for this workstream; the purpose is to replace the current hunch-driven understanding of the 85% dropout with a data-driven one that later phases (FastText retrain, Gemma fine-tune, phrase handling) can be prioritized against.

## Motivation

The valid-anchor rate on the current pipeline is 14.2% (1,965 / 13,886). This is the single cheapest place to recover alignment top-1 — raising the training-anchor count is more leveraged than tuning ridge alpha, which the 2026-04-16 alpha sweep showed is already at its ceiling at 19.85% top-1 for whitened Gemma. But no one in the project knows *why* 85% of anchors drop out. Candidate causes (agglutinative-morphology vocab miss, multi-word English glosses, ORACC→ATF normalization drift, junk filters being overzealous, English target-vocab coverage gaps) are all plausible. Without a categorization, any fix would be speculative.

This audit makes the categorization explicit and reproducible so that any future pipeline change (new anchor extraction, new FastText tokenization, new English target) can be scored by how it moves the bucket distribution. The user's framing is "nail the methodology before tracking insights."

## Scope

### In scope
- New `scripts/audit_anchors.py` — standalone script; reads committed input artifacts, emits dated report pair (md + json).
- New `tests/test_audit_anchors.py` — unit tests against synthetic inputs covering bucket priority, junk detection, English Venn accounting, survival contract, determinism, schema versioning, empty-input edge case.
- New output files under `results/`:
  - `results/anchor_audit_<YYYY-MM-DD>.md` — human-readable report.
  - `results/anchor_audit_<YYYY-MM-DD>.json` — machine-readable (schema version 1).
- Dual-target diagnosis: audit runs against BOTH target English spaces (GloVe 400k, whitened-Gemma 400k) in one invocation and produces a combined report.

### Out of scope
- Any fix to the anchor extraction, FastText training, alignment, or target vocab — strictly diagnostic.
- Modifications to `scripts/06_extract_anchors.py` or any upstream data.
- Decisions about which bucket to tackle first — the recoverability narrative in the report is a paragraph of hand-written heuristics, not an optimization.
- CI integration / pass-fail thresholds.
- Cross-language generalization — Sumerian audit only; audit of a future Akkadian or Egyptian pipeline is a separate spec.

### Deliverables produced
- One new script, one new test file, one new report pair (committed for the 2026-04-18 run as a baseline), one journal entry.

## Success Criteria

- `python scripts/audit_anchors.py` exits 0 against current repo artifacts.
- Report pair is written to `results/anchor_audit_2026-04-18.{md,json}`.
- JSON report sums: `sum(buckets[*].count) == totals.merged`, `buckets.survives.count == totals.survives`, `totals.merged - totals.survives == totals.dropped`. No arithmetic drift.
- `pytest tests/test_audit_anchors.py` — all tests pass.
- `totals.survives` matches the valid-anchor count currently recorded in `final_output/metadata.json` (1,965) to within the tolerance caused by target-space intersection (since the existing `metadata.json` count is computed against GloVe alone; the audit's `survives` requires BOTH GloVe and Gemma, so a small discrepancy is expected and must be documented).
- Two runs with no intervening changes produce byte-identical JSON reports.

## Design

### Architecture

```
data/processed/english_anchors.json     # 13,886 merged anchors (input)
models/fused_embeddings_1536d.npz       # 35,508-word Sumerian vocab (input)
models/english_gemma_whitened_768d.npz  # 400k Gemma vocab (input)
data/processed/glove.6B.300d.txt        # 400k GloVe vocab (input, read lazily — line-by-line, vocab-only, no vectors)
         │
         ▼
   scripts/audit_anchors.py
         │
         ▼
results/anchor_audit_<YYYY-MM-DD>.md
results/anchor_audit_<YYYY-MM-DD>.json
```

The script does NOT re-run alignment, does NOT modify any input, does NOT load vector data (only vocab lists are needed for English-side membership checks — GloVe is read line-by-line taking only the first token per line to minimize memory; Gemma npz only loads the `vocab` array, not `vectors`).

### Bucket taxonomy (mutually exclusive, priority-assigned)

Each anchor is evaluated against the priority-ordered rule chain below. First matching rule determines the bucket; subsequent rules are not evaluated for that anchor.

| Priority | Bucket | Assignment rule |
|---|---|---|
| 1 | `junk_sumerian` | Sumerian side is empty, single character, whitespace, or fails ATF-normalization to any non-empty token |
| 2 | `duplicate_collision` | Anchor's Sumerian key was lost to `merge_anchors` higher-confidence dedup (reconstructed by replaying dedup against the raw ePSD2 + ETCSL outputs) |
| 3 | `low_confidence` | Anchor confidence < 0.3 (matches the current ETCSL co-occurrence threshold in `06_extract_anchors.py`) |
| 4 | `sumerian_vocab_miss` | ATF-normalized Sumerian side not in the 35,508-word fused vocab |
| 5 | `multiword_english` | English side contains a space, hyphen, or underscore after normalization (i.e., is a phrase, not a single token) |
| 6 | `english_both_miss` | Lowercased English side absent from BOTH GloVe AND Gemma vocabs |
| 7 | `english_glove_miss` | Lowercased English side absent from GloVe (but present in Gemma) |
| 8 | `english_gemma_miss` | Lowercased English side absent from Gemma (but present in GloVe) |
| 9 | `survives` | Passes all prior checks — Sumerian side in fused vocab AND English side in both target vocabs |

Priority rationale: junk and dedup collision are "shouldn't have been in the dataset" failures and go first. Low confidence is a threshold decision. Sumerian-side miss is prioritized above English-side misses because it's the harder constraint (one target Sumerian vocab, no fallback) and drives Workstream 2b decisions. Multiword English is separated from English-miss because it's recoverable via a phrase-splitter without changing either target vocab.

### Report contents

#### Markdown (`results/anchor_audit_<YYYY-MM-DD>.md`)

```markdown
# Anchor Audit — YYYY-MM-DD

## Summary
Total merged anchors, surviving count, dropped count — all with percentages.

## Dropout by bucket (priority-assigned, mutually exclusive)
Table with columns: Bucket | Count | % of total | % of dropped | Recoverability.

## Cross-cut: English-side Venn
2×2 Venn of {in GloVe} × {in Gemma} over the set of single-token-English
anchors that passed all priority 1–5 checks (i.e., the denominator equals
`english_both_miss + english_glove_miss + english_gemma_miss + survives`).
Four quadrant counts, confirms the internal consistency of the english_*_miss
buckets above.

## Bucket examples
Ten random rows per bucket (deterministic seed), showing sumerian / english /
confidence / source / a short `notes` column with a human hint about the
likely cause (e.g., "compound token; underscore-split might recover").

## Recoverability narrative
Short paragraph: of the dropped anchors, what fraction is cheap to recover
(Workstream 2a follow-ups), what fraction needs a larger upstream change
(Workstream 2b/2c), what fraction is genuinely untranslatable.
```

#### JSON (`results/anchor_audit_<YYYY-MM-DD>.json`)

```json
{
  "audit_schema_version": 1,
  "audit_date": "YYYY-MM-DD",
  "source_artifacts": {
    "anchors_path": "data/processed/english_anchors.json",
    "anchors_sha256": "…",
    "fused_vocab_path": "models/fused_embeddings_1536d.npz",
    "fused_vocab_size": 35508,
    "glove_path": "data/processed/glove.6B.300d.txt",
    "glove_vocab_size": 400000,
    "gemma_path": "models/english_gemma_whitened_768d.npz",
    "gemma_vocab_size": 400000,
    "seed": 42
  },
  "totals": {
    "merged": 13886,
    "survives": 1965,
    "dropped": 11921
  },
  "buckets": {
    "junk_sumerian":       {"count": …, "pct_total": …, "examples": [ … ]},
    "duplicate_collision": {"count": …, "pct_total": …, "examples": [ … ]},
    "low_confidence":      {"count": …, "pct_total": …, "examples": [ … ]},
    "sumerian_vocab_miss": {"count": …, "pct_total": …, "examples": [ … ]},
    "multiword_english":   {"count": …, "pct_total": …, "examples": [ … ]},
    "english_both_miss":   {"count": …, "pct_total": …, "examples": [ … ]},
    "english_glove_miss":  {"count": …, "pct_total": …, "examples": [ … ]},
    "english_gemma_miss":  {"count": …, "pct_total": …, "examples": [ … ]},
    "survives":            {"count": …, "pct_total": …, "examples": [ … ]}
  },
  "english_venn": {
    "in_glove_in_gemma":     …,
    "in_glove_not_gemma":    …,
    "not_glove_in_gemma":    …,
    "not_glove_not_gemma":   …
  }
}
```

Input-artifact SHA-256s are recorded so a future diff-run can explicitly confirm whether the inputs moved. The `seed: 42` field is fixed — deterministic example selection means diffs across runs show only real bucket-count movement, not example churn.

### Vocab lookup conventions

- **Sumerian side:** reuse `normalize_oracc_cf` from `scripts/06_extract_anchors.py` (import, do not re-implement). Compare against the raw `vocab` array of `models/fused_embeddings_1536d.npz`.
- **English side:** `.lower()` the anchor's `english` field, compare against the lowercased vocab of each target. Both GloVe and Gemma whitened vocabs are already lowercased — audit verifies this in an early startup assert (sample the first 100 tokens, require all-lowercase).
- **Multiword detection:** regex `[\s_\-]` present in the original `english` field (before lowercasing). Anchors whose English field is a single normalized token proceed past this gate.
- **Dedup-collision reconstruction:** re-run `extract_epsd2_anchors` and `extract_cooccurrence_anchors` against the raw `data/raw/oracc_lemmas.json` and `data/raw/etcsl_texts.json`, then compute the set difference between (dict_anchors ∪ cooc_anchors) keyed by Sumerian side, and the merged output keyed the same. Any key present in the pre-merge pool but absent from the post-merge pool is a dedup collision.

### Recoverability heuristics

These live in a constant in the script and are emitted verbatim into the report's `Recoverability` column. They are human judgments, not computations:

```
junk_sumerian         → "low — upstream extraction bug"
duplicate_collision   → "low — dedup by design"
low_confidence        → "medium — raise threshold only if false-positive rate checks out"
sumerian_vocab_miss   → "high — candidate driver for Workstream 2b (FastText retrain with different tokenization)"
multiword_english     → "medium — cheap phrase-splitter or phrase-embedding could recover N%"
english_both_miss     → "low — genuinely untranslatable or specialist vocab"
english_glove_miss    → "medium — check lemmatization / hyphenation variants"
english_gemma_miss    → "medium — same, Gemma-specific"
```

### What the audit does NOT do

- Does not enforce pass/fail thresholds — diagnostic-only.
- Does not modify any input file.
- Does not compute recoverability algorithmically — the narrative paragraph is a human judgment rendered from the heuristic table above.
- Does not run against future target spaces (e.g., fine-tuned Sumerian-Gemma) unless they land in the same path convention; extending to a new target space is a one-line CLI flag addition in a future follow-up.

## Error Handling

| Condition | Behavior |
|---|---|
| Any input artifact missing | `FileNotFoundError` with message pointing to the generating script |
| Raw extraction inputs (`data/raw/oracc_lemmas.json`, `data/raw/etcsl_texts.json`) missing | `duplicate_collision` bucket skipped with a warning; affected anchors fall through to the next matching bucket. Dedup-collision reconstruction requires the raw inputs which are gitignored; absent them, the audit still produces all other buckets. |
| Shape mismatch (fused != 1536d, Gemma != 768d, or GloVe row ≠ 300 floats after token) | `ValueError` with the expected vs actual |
| Anchor JSON malformed (missing `sumerian`, `english`, `confidence`, or `source` on any row) | `ValueError` listing the first 5 bad rows |
| English or Sumerian side is None or non-string | Treated as `junk_sumerian` |
| Target vocab fails all-lowercase sample check | `ValueError` — upstream cache is in unexpected state |
| Duplicate anchor keys in merged input | Raised as `ValueError` — should never happen post-`merge_anchors` |

## Testing Strategy

New file `tests/test_audit_anchors.py`. Tiny synthetic inputs constructed in-test, no dependency on real artifacts. Target <1s total runtime.

- `test_bucket_priority_ordering` — anchor hits `sumerian_vocab_miss` AND `multiword_english` → lands in `sumerian_vocab_miss` (higher priority).
- `test_junk_sumerian_detection` — empty, single-char, whitespace-only all route to `junk_sumerian`.
- `test_english_venn_accounting` — four synthetic anchors (one per Venn quadrant) → counts land in correct cells and sum to 4.
- `test_survives_requires_both_target_vocabs` — English side in GloVe only → `english_gemma_miss`, not `survives`.
- `test_determinism` — two runs with identical inputs → byte-identical JSON.
- `test_report_schema_version` — output has `audit_schema_version: 1`.
- `test_empty_anchors_input` — zero-anchor input → valid report with `totals.merged: 0`, no division errors, all bucket counts 0.

Not tested: the real-data smoke number (the output *is* the tool's value; pinning it in a unit test would fight every pipeline improvement).

## Reproducibility & Provenance

- Dated report filenames capture when the audit ran.
- Input SHA-256 hashes in the JSON report lock down which version of each input produced the result.
- Seed is constant (42). No randomness outside example selection.
- Reports are committed. Future audits produce new timestamped files alongside old ones — history is a directory listing.

## Success paths and failure paths (operational)

**Success path:** developer runs `python scripts/audit_anchors.py`. Script loads three vocabularies (~45s for GloVe), replays dedup against raw inputs (~10s), classifies all 13,886 anchors (<1s), writes two files, prints a one-line summary (`survives: N/13886 (…%), dropped: M`). Exit 0.

**Failure paths:** any missing artifact or malformed input triggers an immediate raise with an actionable message pointing to the upstream generator. No silent partial reports.

## Follow-up work (out of this spec)

The audit's *output* will likely prompt one or more of:
- Phrase-splitter for multi-word English anchors (if `multiword_english` is large).
- Different tokenization in `scripts/05_clean_and_tokenize.py` or FastText training to close `sumerian_vocab_miss` (Workstream 2b).
- Tighter or looser `low_confidence` threshold.
- Upstream filter fix for patterns showing up in `junk_sumerian` that the current regex misses.

Each is a separate brainstorm. The audit provides the numbers they will argue from.
