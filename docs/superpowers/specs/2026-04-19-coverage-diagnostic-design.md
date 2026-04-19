# Sumerian Anchor Coverage Diagnostic (Workstream 2b-pre)

**Date:** 2026-04-19
**Status:** Approved (brainstorming), pending writing-plans
**Branch:** `master` (to be cut into a fresh feature branch at implementation time)
**Follows:** `docs/superpowers/specs/2026-04-18-sumerian-anchor-audit-design.md` (Workstream 2a)
**Journal:** `docs/EXPERIMENT_JOURNAL.md`

## Summary

Build `scripts/coverage_diagnostic.py`: a deeper diagnostic that takes the ~11,798 `sumerian_vocab_miss` anchors from the Workstream 2a baseline audit and tells us *why* each one missed and *which intervention* would recover it. Output is a dated markdown + JSON report (schema v1) containing two independent analyses: a **classifier** (mutually-exclusive primary-cause attribution per anchor) and a **simulator** (per-intervention projected recovery, with Tier-2 semantic validation for inference-based interventions). Fixes are out of scope — this workstream answers "which intervention is actually worth shipping."

## Motivation

The 2026-04-18 audit showed `sumerian_vocab_miss: 11,798 (84.96%)` as the single dominant dropout. The initial framing of the fix was "FastText retrain with different tokenization." An ML-engineer reassessment revealed this is too narrow:

1. The gap is not string-normalization; it is **citation-form ↔ surface-form mismatch**. ORACC anchors are dictionary headwords; the corpus has inflected/compound surface forms. Subscript/hyphen fixes bridge a minority of cases.
2. **FastText's killer feature (character n-gram subword inference) is likely unused** by the current alignment pipeline. Every OOV anchor gets discarded even though FastText could produce a vector for it.
3. **`min_count=5`** is aggressive for a low-resource agglutinative language where many legitimate words occur 1–3 times.
4. **`oracc_lemmas.json` already contains the citation↔surface bridge** — every surface form with its citation form. Anchors could be expanded into their surface variants; the variant in vocab wins.
5. **Hyphenated anchors** are compositional: their morphemes are often individually in FastText vocab even when the compound is not. A morpheme-mean vector is a cheap recovery.

A diagnostic that quantifies each intervention's independent contribution is the right next step. Only then can Workstream 2b (the actual intervention — likely a *portfolio* of small fixes, not one retrain) be scoped correctly.

## Scope

### In scope
- New `scripts/coverage_diagnostic.py` — standalone script; reads committed and gitignored-but-locally-present artifacts; emits dated report pair.
- New `tests/test_coverage_diagnostic.py` — unit tests with tiny synthetic inputs and a toy FastText model trained in-test.
- New output files under `results/`:
  - `results/coverage_diagnostic_<YYYY-MM-DD>.md`
  - `results/coverage_diagnostic_<YYYY-MM-DD>.json` (schema v1)
- Classifier: one of six mutually-exclusive primary causes per missing anchor.
- Simulator: five independent intervention simulators (`ascii_normalize`, `lower_min_count`, `oracc_lemma_expansion`, `morpheme_composition`, `subword_inference`).
- Tier-2 semantic validation: for the two inference-based interventions (`morpheme_composition`, `subword_inference`), project synthesized vectors through the existing whitened-Gemma ridge weights and measure top-K nearest-English-neighbor accuracy against the expected gloss.
- Journal entry covering the baseline diagnostic run.

### Out of scope
- Any actual intervention — no normalization fix, no FastText retrain, no alignment re-run, no anchor file modification.
- Combined-intervention synergy analysis (each intervention reported in isolation).
- Anchor semantic-quality pass (different problem: some of the current 1,951 survives look suspicious; separate future brainstorm).
- Reverse-direction anchor generation (for each FastText vocab word, ask ORACC for an English gloss).
- Cross-language extension (Akkadian/Egyptian pipelines).
- CI integration or pass/fail thresholds — diagnostic-only.

### Deliverables produced
- One new script, one new test file, one report pair (committed), one journal entry.

## Success Criteria

- `python scripts/coverage_diagnostic.py` exits 0 against current repo artifacts.
- Report pair is written to `results/coverage_diagnostic_2026-04-19.{md,json}`.
- JSON report arithmetic:
  - `sum(classifier.primary_causes[*].count) == classifier.total_misses`
  - `classifier.total_misses` matches `audit.buckets.sumerian_vocab_miss.count` from the 2026-04-18 audit (11,798) within tolerance for anchor-file SHA drift (if upstream anchors changed, the number changes; the SHA stamp records the input state).
  - For each intervention in the simulator, `anchors_newly_resolvable >= 0` and `projected_survives >= baseline.survives`.
- `pytest tests/test_coverage_diagnostic.py` — all tests pass.
- Two consecutive runs produce byte-identical JSON reports.

## Design

### Architecture

```
results/anchor_audit_2026-04-18.json         (source of misses; recomputed internally for decoupling)
data/processed/english_anchors.json
data/processed/cleaned_corpus.txt             (min_count frequency analysis)
data/processed/glove.6B.300d.txt              (English vocab for Tier-2 projection)
data/raw/oracc_lemmas.json                    (REQUIRED — not optional, unlike audit)
models/fasttext_sumerian.model                (full FastText object, needed for subword)
models/fused_embeddings_1536d.npz             (explicit vocab check)
models/english_gemma_whitened_768d.npz
models/ridge_weights_gemma_whitened.npz       (Tier-2 projection target)
models/ridge_weights.npz                      (optional: Tier-2 via GloVe for completeness)
         │
         ▼
   scripts/coverage_diagnostic.py
         │
         ▼
results/coverage_diagnostic_<YYYY-MM-DD>.md
results/coverage_diagnostic_<YYYY-MM-DD>.json
```

The script operates ONLY on the `sumerian_vocab_miss` anchors. It re-runs the audit's classifier internally (reusing helpers from `scripts/audit_anchors.py`) to identify misses — no coupling to the previous audit JSON. This self-containment means the diagnostic is valid against whatever the current artifacts say, not a stale snapshot.

### Classifier

For each missing anchor, evaluate checks in priority order (first match wins). Priorities encode trustworthiness — exact matches rank above inferred vectors.

```
PRIMARY_CAUSE_ORDER = (
    "normalization_recoverable",           # apply 05-equivalent normalization -> exact vocab hit
    "in_corpus_below_min_count",           # exact surface form in cleaned_corpus.txt, count < 5
    "oracc_lemma_surface_recoverable",     # ORACC has citation -> surface mapping; surface in vocab
    "morpheme_composition_recoverable",    # hyphenated; ALL morphemes in vocab
    "subword_inference_recoverable",       # FastText subword n-gram overlap >= threshold
    "genuinely_missing",                   # none of the above
)
```

The classifier's primary-cause counts sum to `total_misses` exactly. Each bucket's JSON entry includes 10 deterministically-sampled example rows with a `trace` field describing the match (e.g., `{"normalized_form": "tug2mug"}` or `{"matched_surface_form": "nar.ta"}` or `{"morphemes_in_vocab": ["za3", "sze3", "la2"]}`).

### Normalization rule

Mirror the chain in `scripts/05_clean_and_tokenize.py`:
1. Unicode subscript digits → ASCII digits.
2. Strip determinative braces `{...}` keeping the content.
3. ORACC unicode Sumerian letters → ATF (reuse existing `normalize_oracc_cf` map: `š → sz`, `ḫ → h`, etc.).
4. For hyphenated tokens, check both the fully-joined compound form (`za3-sze3-la2` → `za3sze3la2`) AND the raw form; either match counts.
5. Lowercase.

The classifier reuses the same normalization function for the `normalization_recoverable` check and exposes it to both `05_clean_and_tokenize.py` (as a future consolidation — out of scope for this spec) and `06_extract_anchors.py` (also future).

### Simulator

Five simulators run independently. Each answers: "if I shipped *only* this intervention, how many additional anchors would be resolvable?" Counts CAN overlap (an anchor may be recoverable by multiple interventions) — that is the complementary function to the classifier's mutually-exclusive attribution.

#### 1. `ascii_normalize`
- Apply the normalization chain above to each OOV anchor.
- Count anchors whose normalized form is in the explicit FastText vocab.
- Trustworthiness: exact.
- No Tier-2 (exact match → vector is the trained FastText vector).

#### 2. `lower_min_count`
- Build a frequency table from `cleaned_corpus.txt` (count occurrences per token).
- For each OOV anchor, look up its occurrence count.
- Report per-threshold (1, 2, 3, 4) how many new anchors would become resolvable if `min_count` were that value. Assumes a FastText retrain would be required to actually ship this.
- Trustworthiness: exact (at the stated threshold, the form would be in the retrained vocab).
- No Tier-2 (we're not retraining to measure top-K).

#### 3. `oracc_lemma_expansion`
- Load `data/raw/oracc_lemmas.json`. Build a map `citation_form -> {surface_forms}`.
- For each OOV anchor, collect its citation-form's surface forms; check if any is in explicit FastText vocab.
- Count anchors with at least one in-vocab surface variant. Report also `surface_forms_added_to_vocab` (total unique surface forms added across all recovered anchors).
- Trustworthiness: exact.
- No Tier-2 (exact match via surface form).

#### 4. `morpheme_composition`
- For each OOV anchor containing hyphens, split into morphemes.
- Check if ALL morphemes are in explicit FastText vocab.
- If yes, synthesize a vector as the numpy mean of the morpheme vectors.
- **Tier 1:** count anchors where all morphemes are in vocab.
- **Tier 2:** for each Tier-1 recoverable anchor whose English side IS in the whitened-Gemma vocab, project the synthesized Sumerian vector through `ridge_weights_gemma_whitened.npz` + apply Gemma-space cosine nearest neighbor. Report top-1/top-5/top-10 correctness (expected English = anchor's gloss).
- Trustworthiness: inferred (compositional).

#### 5. `subword_inference`
- Load the full `fasttext_sumerian.model`.
- For each OOV anchor, compute character n-grams (using model's `min_n`/`max_n`).
- **Recoverable** = at least `SUBWORD_OVERLAP_MIN = 0.5` of the anchor's n-grams are present in the trained n-gram table.
- **Tier 1:** count anchors meeting the overlap threshold.
- **Tier 2:** for each Tier-1 recoverable anchor, get `ft.wv.get_vector(anchor)`, fuse with zero-padding to 1536d (matching `scripts/08_fuse_embeddings.py`), project through `ridge_weights_gemma_whitened.npz`, check nearest English neighbor. Report top-1/top-5/top-10.
- Trustworthiness: inferred (character n-gram).

### Report structure

#### JSON (`results/coverage_diagnostic_<YYYY-MM-DD>.json`)

```json
{
  "diagnostic_schema_version": 1,
  "diagnostic_date": "YYYY-MM-DD",
  "source_artifacts": {
    "anchors_path": "...",
    "anchors_sha256": "...",
    "fasttext_model_path": "...",
    "fasttext_model_sha256": "...",
    "fused_vocab_path": "...",
    "glove_path": "...",
    "gemma_path": "...",
    "ridge_gemma_path": "...",
    "oracc_lemmas_path": "...",
    "cleaned_corpus_path": "...",
    "cleaned_corpus_sha256": "...",
    "seed": 42,
    "subword_overlap_min": 0.5
  },
  "baseline": {
    "total_merged": 13886,
    "survives": 1951,
    "sumerian_vocab_miss": 11798
  },
  "classifier": {
    "total_misses": 11798,
    "primary_causes": {
      "normalization_recoverable":        {"count": N, "pct": X, "examples": [...]},
      "in_corpus_below_min_count":        {"count": N, "pct": X, "examples": [...]},
      "oracc_lemma_surface_recoverable":  {"count": N, "pct": X, "examples": [...]},
      "morpheme_composition_recoverable": {"count": N, "pct": X, "examples": [...]},
      "subword_inference_recoverable":    {"count": N, "pct": X, "examples": [...]},
      "genuinely_missing":                {"count": N, "pct": X, "examples": [...]}
    }
  },
  "simulator": {
    "interventions": {
      "ascii_normalize":        {"anchors_newly_resolvable": N, "projected_survives": N, "projected_pct": X, "trustworthiness": "exact"},
      "lower_min_count":        {"per_threshold": {"1": {...}, "2": {...}, "3": {...}, "4": {...}}, "trustworthiness": "exact"},
      "oracc_lemma_expansion":  {"anchors_newly_resolvable": N, "surface_forms_added_to_vocab": N, "projected_survives": N, "trustworthiness": "exact"},
      "morpheme_composition":   {"anchors_newly_resolvable_tier1": N, "projected_survives_tier1": N, "tier2_semantic": {"tested": N, "top1_correct": N, "top5_correct": N, "top10_correct": N}, "projected_survives_tier2_top5": N, "trustworthiness": "inferred (compositional)"},
      "subword_inference":      {"anchors_newly_resolvable_tier1": N, "projected_survives_tier1": N, "tier2_semantic": {"tested": N, "top1_correct": N, "top5_correct": N, "top10_correct": N}, "projected_survives_tier2_top5": N, "trustworthiness": "inferred (character n-gram)"}
    }
  }
}
```

#### Markdown (`results/coverage_diagnostic_<YYYY-MM-DD>.md`)

Sections:
- `# Coverage Diagnostic — YYYY-MM-DD`
- `## Baseline` (survives, total misses, what we're analyzing)
- `## Classifier — primary-cause attribution` (bucket table, pct of misses, pct of total, examples per bucket)
- `## Simulator — per-intervention projected recovery` (ranked table by projected_survives descending; Tier-2 interventions show both Tier-1 and Tier-2 numbers)
- `## Ranked intervention recommendations` (one-paragraph narrative: given these numbers, ship interventions in order X, Y, Z with rationale)
- `## Methodology notes` (brief: what "exact" vs "inferred" means; why Tier-2 validation; SHA stamps)

### Determinism

- Example row selection uses `np.random.default_rng(seed=42)`.
- All iteration orders are deterministic (sorted keys).
- Two consecutive runs against the same inputs (same SHAs) produce byte-identical JSON.

### SHA-stamping

All input artifacts (anchors, cleaned corpus, FastText model file) hash-stamped into the JSON report. Future diffs can diagnose "did this number change because the intervention moved, or because upstream data changed?"

## Error Handling

| Condition | Behavior |
|---|---|
| Any required input missing | `FileNotFoundError` with message pointing to the generating script. |
| `data/raw/oracc_lemmas.json` missing | `FileNotFoundError`. This is REQUIRED (unlike the audit's graceful skip), because `oracc_lemma_expansion` is a core intervention. Message points to `scripts/03_scrape_oracc.py`. |
| `models/fasttext_sumerian.model` missing or corrupt | `FileNotFoundError` / `RuntimeError` — subword and morpheme interventions both need it. Message points to `scripts/07_train_fasttext.py`. |
| Tier-2: OOV anchor's English side not in Gemma vocab | Silently excluded from `tested`; reported as `skipped: N` in the simulator output. |
| Morpheme composition on non-hyphenated anchor | Not a candidate; doesn't count against `anchors_newly_resolvable`. |
| Anchor normalizes to empty string | Classified as `genuinely_missing`. |
| Anchor whose Sumerian side is None or malformed | Treated as `genuinely_missing` (defensive). |
| Duplicate anchor keys in input | Raises `ValueError` — should never happen post-`merge_anchors`. |
| Ridge weights file shape mismatch | `ValueError` with expected-vs-actual dim. |

## Testing Strategy

New `tests/test_coverage_diagnostic.py`. Synthetic inputs, toy FastText trained in-test. Target <5s total runtime.

### Classifier tests
- `test_normalization_wins_over_subword` — priority ordering holds.
- `test_below_min_count_wins_over_lemma` — priority ordering holds.
- `test_morpheme_composition_requires_all_morphemes_in_vocab` — 3 morphemes, 2 in vocab → NOT recoverable; 3/3 → recoverable.
- `test_subword_threshold_enforced` — below 50% overlap → genuinely_missing.
- `test_genuinely_missing_catchall` — anchor matching no intervention → genuinely_missing.

### Simulator tests
- `test_simulator_ascii_normalize_count`
- `test_simulator_lower_min_count_per_threshold` — monotone non-increasing across thresholds 1→4.
- `test_simulator_oracc_lemma_expansion_counts_unique_anchors` — one citation form with 3 surface variants (2 in vocab) counts as 1 recovered anchor, not 2.
- `test_simulator_morpheme_composition_vector_is_mean` — exact numpy-level check.
- `test_simulator_subword_inference_uses_wv_get_vector` — FastText OOV produces non-zero vector; simulator counts correctly.

### Tier-2 semantic validation tests
- `test_tier2_projects_through_ridge_weights` — synthetic ridge + English vocab; Tier-2 identifies correct English neighbor.
- `test_tier2_skips_anchors_without_english_in_vocab` — reported as skipped, not counted against tested.

### Report rendering tests
- `test_render_json_schema_version_1`
- `test_render_markdown_has_required_sections`
- `test_determinism` — byte-identical rerun against the same synthetic inputs.

### Integration smoke (optional, `@pytest.mark.slow`)
- `test_full_diagnostic_against_real_artifacts` — runs end-to-end against the real repo artifacts if locally present; skipped on fresh clones.

### Fixture helpers
- `_train_toy_fasttext(tmpdir, corpus, min_count=1, vector_size=32)` — shared toy-model fixture.

## Reproducibility & Provenance

- Dated report filenames.
- Input SHA-256s in the JSON report.
- Fixed seed (42) for all randomness.
- Methodology constants (`SUBWORD_OVERLAP_MIN = 0.5`) in module-level constants, not CLI flags — methodology stability over flexibility.
- Reports are committed. Future diagnostics produce new timestamped files alongside old ones.

## Operational notes

**Success path:** developer runs `python scripts/coverage_diagnostic.py`. Script loads artifacts (~1 min for GloVe and FastText model), runs audit classifier to identify misses (~1s), runs priority-ordered classifier (~1s), runs 5 simulators (longest: Tier-2 projections, ~10–30s depending on vocab sizes), writes two files, prints summary. Exit 0.

**Failure paths:** any missing artifact triggers an immediate actionable raise. No partial reports.

## Follow-up work (out of this spec)

The diagnostic's *output* directly scopes the next brainstorm(s):
- **Workstream 2b (scope TBD by this diagnostic's output):** likely a portfolio of small interventions. Possibilities include shared-normalization consolidation, ORACC surface-form anchor expansion, lowering `min_count`, integrating FastText subword inference at anchor lookup time (no retrain needed). Each is a candidate for its own spec or a combined short sprint.
- **Workstream 2b retrain (contingent):** only if the diagnostic shows a significant `genuinely_missing` bucket that a tokenization change would help. Not a default next step.
- **Survives-quality pass (separate, later):** the current 1,951 survives includes some suspicious entries (names, single-letter glosses). Orthogonal to coverage; probably worth a brainstorm after 2b.

Each is its own spec → plan → implementation cycle.
