# Workstream 2b: Sumerian Anchor Normalization Fix

**Date:** 2026-04-19
**Status:** Approved (brainstorming), pending writing-plans
**Branch:** `master` (to be cut into a fresh feature branch at implementation time)
**Follows:** `docs/superpowers/specs/2026-04-19-coverage-diagnostic-design.md` (Workstream 2b-pre)
**Journal:** `docs/EXPERIMENT_JOURNAL.md`

## Summary

Close the trivial unicode-normalization gap between ORACC citation forms (anchor side) and ATF surface forms (corpus side) by applying the canonical normalization chain — subscripts → ASCII, strip determinative braces, ORACC → ATF letters, drop hyphens, lowercase — to every anchor's Sumerian side at extraction time. Ship as a ~20-line code change plus a rerun of `06 → 09/09b → 10`, committed as a single focused workstream with measurable top-1 and valid-anchor deltas.

## Motivation

The 2026-04-19 coverage diagnostic (commit `5fb0ca3`) attributed **64.85% of the `sumerian_vocab_miss` bucket (7,651 of 11,798 anchors)** to `normalization_recoverable`: anchors whose Sumerian side becomes an *exact* FastText vocab hit after the canonical normalization chain is applied. `scripts/06_extract_anchors.py::normalize_oracc_cf` currently applies only the ORACC letter map (`š → sz`, etc.); it omits subscript conversion, brace stripping, and hyphen handling — all three of which `scripts/05_clean_and_tokenize.py` already applies to the corpus. Anchors and corpus live in different normalization spaces, which is the root of the ~85% anchor dropout.

Every alternative intervention measured (subword inference, morpheme composition, lower min_count, ORACC lemma surface expansion) was smaller; the two inference-based ones were also validated at low Tier-2 semantic accuracy (10.7% and 1.8% top-5). Normalization is both the biggest and the cleanest lever.

Phase B's current best top-1 is 19.85% (whitened Gemma, 1,572 training anchors). Heiroglyphy at ~5,360 training anchors reaches 32.35% top-1. Adding ~7,000 clean training anchors post-normalization is expected to lift top-1 meaningfully (honest range: 25–35%).

## Scope

### In scope
- New shared module `scripts/sumerian_normalize.py` containing the canonical token normalization function + its supporting constants.
- New unit tests `tests/test_sumerian_normalize.py`.
- One new test case added to `tests/test_06_anchors.py` verifying the extractor applies the full normalization.
- Modification to `scripts/06_extract_anchors.py` to use the shared module's function in place of its local `normalize_oracc_cf`.
- Modification to `scripts/coverage_diagnostic.py` to import from the shared module (eliminating the duplicate local definition).
- Minor refactor of `tests/test_coverage_diagnostic.py` to rename `normalize_anchor_form` imports to `normalize_sumerian_token`.
- Pipeline rerun: `06 → 09 → 09b → 10`, plus regression-check reruns of `validate_phase_b.py`, `audit_anchors.py`, `coverage_diagnostic.py`.
- Single consolidated commit of regenerated pipeline artifacts.
- Journal entry recording the real top-1 + audit delta.

### Out of scope
- `scripts/05_clean_and_tokenize.py` refactor to use the shared module. Out-of-scope DRY win; risks regressions in corpus tokenization.
- Retraining FastText (no corpus change).
- Retraining the Gemma whitening transform (target space unchanged).
- `lower_min_count` tuning (separate workstream, if warranted by residual diagnostic).
- Anchor semantic-quality pass — removing noisy anchors like `sirara→c` (orthogonal, separate brainstorm).
- Any target-space additions or changes.

### Deliverables produced
- One new module, one new test file, three modified scripts, one modified test file, one new test case in an existing test file.
- Regenerated pipeline artifacts under `final_output/`, `models/`, `results/`.
- A dated 2026-04-19 or later entry in `docs/EXPERIMENT_JOURNAL.md`.

## Success Criteria

Acceptance is tiered by measured top-1 on whitened-Gemma alignment (run on exactly the same anchors-to-train/test split methodology as prior Phase A/B runs):

| Tier | Top-1 | Action |
|---|---|---|
| **Blocker** | < 19.85% | Ridge regressed. Debug (noisy anchors, bug in normalization); do NOT commit regenerated artifacts until resolved. |
| **Ship target** | ≥ 22% (+2pp) | Ship. Commit regenerated artifacts, journal, merge to master. Queue follow-ups based on residual diagnostic. |
| **Stretch** | ≥ 25% (+5pp) | Ship and flag in the journal as confirmation of the 2b-pre prediction. |

Secondary regression checks (all required, not just acceptance-tiered):

- `audit_anchors.py`'s `sumerian_vocab_miss` count drops from 11,798 toward ~4,147 (= 11,798 − 7,651 ± small slack for English-side filtering).
- `audit_anchors.py`'s `survives` count rises from 1,951 toward ~9,100 (= 1,951 + 7,651 − small slack for English-side filters on the newly-recovered anchors).
- `coverage_diagnostic.py`'s `normalization_recoverable` bucket drops to near 0 (ideally < 100). If it stays high, the new code didn't take effect — debug the wiring.
- `validate_phase_b.py` exits 0; regenerated `concept_clusters_comparison_whitened.md` is structurally identical (same domain headers, similar ranking).
- All unit tests pass: `pytest tests/ --ignore=tests/test_integration.py` expected to show 110 + new tests passing, 0 regressions.

## Design

### Shared module: `scripts/sumerian_normalize.py`

```python
"""
Canonical Sumerian token normalization.

Single source of truth for mapping ORACC citation forms and inflected surface
forms to the common ATF-based token form produced by scripts/05_clean_and_tokenize.py.

Used by scripts/06_extract_anchors.py (anchor side) and
scripts/coverage_diagnostic.py (audit/diagnostic side). Keeping this function
in one place prevents normalization drift between anchors and corpus.

See: docs/superpowers/specs/2026-04-19-workstream-2b-normalization-fix-design.md
"""
from __future__ import annotations

import re

_SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

_ORACC_TO_ATF = {
    "š": "sz", "Š": "SZ",
    "ŋ": "j",  "Ŋ": "J",
    "ḫ": "h",  "Ḫ": "H",
    "ṣ": "s",  "Ṣ": "S",
    "ṭ": "t",  "Ṭ": "T",
    "ʾ": "",
}

_BRACE_RE = re.compile(r"\{([^}]*)\}")


def normalize_sumerian_token(raw) -> str:
    """Canonical normalization for a single Sumerian token.

    Applies (in order):
      1. Unicode subscript digits -> ASCII digits
      2. Strip determinative braces {X} keeping content
      3. ORACC Sumerian unicode letters -> ATF (š -> sz, etc.)
      4. Drop hyphens (produces fully-joined compound form)
      5. Lowercase + strip whitespace

    Safe on None and empty input (returns "").
    Idempotent: normalize(normalize(x)) == normalize(x).
    """
    if raw is None:
        return ""
    s = str(raw)
    s = s.translate(_SUBSCRIPT_MAP)
    s = _BRACE_RE.sub(r"\1", s)
    for old, new in _ORACC_TO_ATF.items():
        s = s.replace(old, new)
    s = s.replace("-", "")
    return s.lower().strip()
```

### Integration: `scripts/06_extract_anchors.py`

- Delete local `_ORACC_TO_ATF` dict and `normalize_oracc_cf` function body.
- Add `from scripts.sumerian_normalize import normalize_sumerian_token`.
- Replace both call sites in `extract_epsd2_anchors` (on `cf` and `form`) with `normalize_sumerian_token(...)`.
- Keep the sys.path guard pattern already used by other scripts if needed for direct invocation.

### Integration: `scripts/coverage_diagnostic.py`

- Delete local `_SUBSCRIPT_MAP`, `_ORACC_TO_ATF`, `_normalize_oracc_to_atf`, `normalize_anchor_form` definitions.
- Add `from scripts.sumerian_normalize import normalize_sumerian_token`.
- Rename every internal use of `normalize_anchor_form` to `normalize_sumerian_token`. (The function name change is load-bearing for clarity — existing behavior preserved.)
- `_morphemes` remains local; it uses `normalize_sumerian_token` on each morpheme piece.

### Pipeline execution

1. `python scripts/06_extract_anchors.py` — regenerates `data/processed/english_anchors.json`. Expected: similar total anchor count, ~7,000 more Sumerian-side hits against FastText vocab.
2. `python scripts/09_align_and_evaluate.py` — regenerates `models/ridge_weights.npz` + `results/alignment_results.json`. Expected: training_anchors up ~5×, top-1 ≥ 22%.
3. `python scripts/09b_align_gemma.py --mode whitened` — regenerates `models/ridge_weights_gemma_whitened.npz` + `results/alignment_results_gemma_whitened.json`. Same expectation.
4. `python scripts/10_export_production.py` — regenerates both `final_output/sumerian_aligned_*.npz` plus `final_output/metadata.json` (schema v2).
5. `python scripts/validate_phase_b.py` — must exit 0.
6. `python scripts/audit_anchors.py --date <today>` — regenerates audit report pair.
7. `python scripts/coverage_diagnostic.py --date <today>` — regenerates coverage diagnostic report pair.

### Artifact commit policy

Regenerated artifacts are committed in a single `chore:` commit, distinct from the code-change `feat:` commit. This keeps the code change diff minimal and the pipeline-artifact delta auditable.

Artifacts:
- `data/processed/english_anchors.json` — gitignored; NOT committed.
- `models/ridge_weights.npz`, `models/ridge_weights_gemma_whitened.npz` — gitignored; NOT committed.
- `final_output/metadata.json` — committed.
- `final_output/sumerian_aligned_*.npz`, `final_output/sumerian_aligned_vocab.pkl` — gitignored; NOT committed.
- `results/alignment_results.json`, `results/alignment_results_gemma_whitened.json` — gitignored; NOT committed (they're local run artifacts per existing repo convention).
- `results/anchor_audit_<date>.{md,json}` — committed (force-added, matching Workstream 2a pattern).
- `results/coverage_diagnostic_<date>.{md,json}` — committed (force-added, matching Workstream 2b-pre pattern).
- `results/concept_clusters_comparison_whitened.md` — committed (regenerated by validate_phase_b).

## Error Handling

| Condition | Behavior |
|---|---|
| `normalize_sumerian_token(None)` | Returns `""`. |
| `normalize_sumerian_token("")` | Returns `""`. |
| Input contains unexpected unicode outside the ORACC → ATF map | Left alone; downstream vocab check simply won't match (anchor falls into `sumerian_vocab_miss` for a different reason). Not a regression vs current behavior. |
| Post-normalization empty string | Anchor is emitted with `sumerian=""` which the audit's `classify_anchor` treats as `junk_sumerian` — same as before. |
| Post-normalization anchor collides with an existing anchor on the Sumerian key | `merge_anchors` already dedups by Sumerian key keeping the higher-confidence entry. No code change needed. |
| Top-1 regresses below 19.85% after the pipeline rerun | Blocker. Do NOT commit regenerated artifacts. Debug the normalization or investigate anchor quality before proceeding. |

## Testing Strategy

### `tests/test_sumerian_normalize.py` (new)

Unit tests for the canonical function. All use tiny literal inputs.

- `test_subscripts_to_ascii` — all 10 unicode subscripts → ASCII digits.
- `test_strips_determinative_braces` — `{tug₂}mug` → `tug2mug`; `{d}en-lil` → `denlil`.
- `test_oracc_to_atf_letters` — each of the 11 map entries (both lower and upper cases).
- `test_drops_hyphens` — `za₃-sze₃-la₂` → `za3sze3la2`; `nar-ta` → `narta`.
- `test_lowercases` — `LUGAL` → `lugal`.
- `test_strips_whitespace` — `" lugal "` → `lugal`.
- `test_handles_empty_and_none` — `""` and `None` → `""`.
- `test_idempotent` — `f(f(x)) == f(x)` for 5 varied inputs.
- `test_combined_chain` — `{Tug₂}-Sze₃-la₂` → `tug2sze3la2`.

### `tests/test_06_anchors.py` (new test added)

- `test_extract_epsd2_anchors_applies_full_normalization` — synthetic lemma with a hyphenated, subscripted, braced Sumerian side produces an anchor whose `sumerian` field matches the expected normalized form.

### `tests/test_coverage_diagnostic.py` (modified)

- Every import of `normalize_anchor_form` renamed to `normalize_sumerian_token`.
- Every direct call site renamed consistently.
- Test `test_normalize_anchor_form_handles_subscripts_braces_and_oracc` renamed to `test_normalize_sumerian_token_handles_subscripts_braces_and_oracc`.
- Behavior assertions unchanged; only identifier rename.

## Reproducibility

- The shared function is deterministic and pure.
- Two consecutive pipeline reruns with no upstream changes produce byte-identical artifacts.
- The audit and coverage diagnostic already SHA-stamp their input artifacts, so before/after comparisons attribute cleanly to this change.

## Operational notes

**Success path:** developer runs the 7-step pipeline rerun, observes the expected top-1 and audit-survives deltas, commits two separate commits (feat: + chore:), updates the journal, merges to master, pushes to origin.

**Failure paths and diagnostics:**

- Top-1 drops below 19.85%: the ridge fit has regressed. Likely cause — some of the newly-recovered 7,651 anchors are semantically wrong (e.g., citation form `X` coincidentally normalizes to an unrelated corpus token `Y`). Debug: compare top-1 on the intersection of old-and-new anchor sets vs the new-only subset.
- Audit `sumerian_vocab_miss` doesn't drop: the normalization isn't being applied where expected. Debug: inspect `english_anchors.json` before/after — the Sumerian fields should now contain no subscripts, braces, or hyphens.
- `coverage_diagnostic.py`'s `normalization_recoverable` bucket stays non-zero: `06` is still applying the old normalization somewhere. Debug: grep for remaining references to the old ORACC map.

## Follow-up work (out of this spec)

Scoped by the residual after 2b ships:

- **Workstream 2c (contingent):** lower `min_count` in FastText training. Diagnostic projected 926 anchors at threshold=1, diminishing to 218 at threshold=4. Requires an actual FastText retrain. Only worth it if 2b's top-1 plateaus below the stretch target.
- **Anchor semantic-quality pass (separate brainstorm):** some current `survives` are suspect (`sirara→c`, `er3→erra`). A quality filter that removes or down-weights low-signal anchors is orthogonal to coverage and belongs after 2b.
- **`subword_inference` / `morpheme_composition`:** Tier-2 top-5 accuracy was 10.7% and 1.8% in the 2b-pre diagnostic. Not worth pursuing without further research on how to make synthesized vectors useful.
