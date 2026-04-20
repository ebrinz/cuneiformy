# Sumerian Cosmogony: A Geometric Translation (document + analysis infrastructure)

**Date:** 2026-04-19
**Status:** Approved (brainstorming), pending writing-plans
**Branch:** `master` (to be cut into a fresh feature branch at implementation time)
**Follows:** `docs/superpowers/specs/2026-04-19-workstream-2b-normalization-fix-design.md` (Workstream 2b)
**Journal:** `docs/EXPERIMENT_JOURNAL.md`

## Summary

Produce `docs/sumerian_cosmogony.md`: a ~10,500-word methodology-rigorous case study that uses the Cuneiformy whitened-Gemma alignment to do "geometric translation" of Sumerian cosmogonic vocabulary, with the Anunnaki as grounding cast. The document pairs a narrative spine (Nammu → An/Ki separation → Anunnaki → Enki shapes humans → destinies decreed → me distributed) with five concept deep dives (`abzu`, `zi`, `nam`, `nam-tar`, `me`) each following an 8-section paper-grade template. Supporting analysis infrastructure (`scripts/analysis/`) is testable, reproducible, and consumed by the document via committed figures and a canonical `cosmogony_tables.json`.

## Motivation

Cuneiformy's infrastructure was built for this: `RESEARCH_VISION.md` frames the project's scientific goal as Riemannian geometry of semantic drift across civilizational time; `NEAR_TERM_STRATEGY.md` names three Phase-1 domains (origin/creation, fate/meaning, self/soul) and Phase-3 "narrative extraction" as the first shippable research artifact. Post Workstream 2b, the whitened-Gemma alignment at 52.13% top-1 is more than strong enough to support geometric interpretation — well past the "is this signal or noise?" threshold.

The Anunnaki are the right grounding cast because their cosmological function — setting destinies, distributing `me` (cultural blueprints), shaping humans from clay — touches every concept in our deep-dive slate. The narrative provides a readable arc; the concepts provide analytical depth; the combination lets a vector-space-literate reader see what geometric translation can do and what it cannot claim.

## Audience

Interdisciplinary research readers at the A/B midpoint:

- **Assyriologists and Sumerologists:** get rigorous methodology (explicit pipeline description, ETCSL citations, acknowledged corpus limitations, calibration against Jacobsen/Kramer/Black). Can critique the method and the specific claims.
- **AI/NLP/interp researchers:** get a concrete example of what semantic alignment reveals when pushed on cosmogonically-dense vocabulary. Vector-space operations (nearest neighbors, cluster distances, vector displacement, analogies) are shown in use rather than hidden.

Requires vector-space familiarity but not Sumerological background. ETCSL citations explained for non-specialists on first use.

## Scope

### In scope

- New document `docs/sumerian_cosmogony.md` (~10,500 words of prose, ~15-18 printed pages).
- 5 concept deep dives following an 8-section template each: `abzu`, `zi`, `nam`, `nam-tar`, `me`.
- Narrative spine (§3): cosmogonic arc from Nammu to the distribution of `me`, written to link the deep dives.
- Synthesis section (§9): what the five concept findings collectively reveal; projection of concepts onto a cosmogonic axis figure; explicit limits of the method.
- New `scripts/analysis/` directory with 6 modules (`semantic_field`, `english_displacement`, `etcsl_passage_finder`, `umap_projection`, `preflight_concept_check`) plus two top-level regeneration entry points (`generate_cosmogony_figures.py`, `generate_cosmogony_tables.py`).
- 7 committed PNG figures under `docs/figures/cosmogony/`.
- 1 canonical JSON file `docs/cosmogony_tables.json` holding every numeric table referenced in the prose.
- Pre-flight concept-availability check with explicit substitution rules if any candidate concept fails.
- Test coverage for all new analysis modules against synthetic inputs.
- Journal entry upon completion with reproducibility pointer.

### Out of scope

- Any new alignment work — the document uses the current 52%-top-1 whitened-Gemma artifacts as-is.
- Egyptian comparison or cross-civilizational analysis — separately-queued new repo.
- `RESEARCH_VISION.md` Phase 2 Riemannian geometry (curvature tensors, distortion maps) — heavy differential geometry deferred.
- Interactive visualizations, web dashboards, or Plotly/UMAP explorers — static PNGs only.
- Steering / interp bridge work — separate research track.
- FastText retraining, corpus expansion, anchor semantic-quality pass.
- Publication venue decisions beyond "it's a committed markdown file."

### Deliverables produced

- 1 new document, 6 new analysis modules, 2 new top-level entry-point scripts, 7 new figures, 1 new JSON tables file, 6 new test files, 1 journal entry.

## Success Criteria

- All 5 deep dives (§4-8) have all 8 template sub-sections filled; no TODOs in prose.
- All 7 figures exist as committed PNGs.
- `docs/cosmogony_tables.json` regenerates byte-identically from `generate_cosmogony_tables.py` (determinism seed + sorted iteration).
- Every numeric claim in prose traces to a row in `cosmogony_tables.json` OR the pre-flight JSON.
- Every cosmogonic or Sumerological claim cites either an ETCSL text (with text-ID) or a named secondary source.
- `pytest tests/analysis/` — all new tests pass.
- `pytest tests/ --ignore=tests/test_integration.py` — no regressions in existing 120 tests.
- Pre-flight appendix (§12) shows which of the 5 concepts were validated; substitutions documented.
- Document's §10 names the pinned commit whose alignment artifacts produced the cited numbers.

## Design

### Architecture and file layout

```
docs/
  sumerian_cosmogony.md                    NEW — main document
  cosmogony_tables.json                    NEW — canonical numeric tables, regenerable
  figures/cosmogony/                       NEW — committed PNGs
    anunnaki_narrative_umap.png
    nam-tar_semantic_field_heatmap.png
    me_semantic_field_heatmap.png
    nam_semantic_field_heatmap.png
    abzu_semantic_field_heatmap.png
    zi_semantic_field_heatmap.png
    cosmogony_axis_projection.png

scripts/analysis/                          NEW
  __init__.py
  semantic_field.py
  english_displacement.py
  etcsl_passage_finder.py
  umap_projection.py
  preflight_concept_check.py
  generate_cosmogony_figures.py            (entry)
  generate_cosmogony_tables.py             (entry)

tests/analysis/                            NEW
  __init__.py
  test_semantic_field.py
  test_english_displacement.py
  test_etcsl_passage_finder.py
  test_umap_projection.py
  test_preflight_concept_check.py

results/
  cosmogony_preflight_<YYYY-MM-DD>.json    NEW — pre-flight report, committed
```

### Document structure (`docs/sumerian_cosmogony.md`)

```
§ 0   Abstract                                  (~150 words)
§ 1   Introduction                              (~500 words)
§ 2   Methodology                               (~800 words)
§ 3   The Cosmogonic Arc (narrative spine)      (~1,500 words + 1 UMAP figure)
§ 4   Deep dive — abzu (primordial deep)        (~1,600 words + 1 heatmap)
§ 5   Deep dive — zi (breath/life-essence)      (~1,600 words + 1 heatmap)
§ 6   Deep dive — nam (essence/office)          (~1,600 words + 1 heatmap)
§ 7   Deep dive — nam-tar (fate/destiny)        (~1,600 words + 1 heatmap)
§ 8   Deep dive — me (divine decrees)           (~1,600 words + 1 heatmap)
§ 9   Synthesis: cosmogony as geometric object  (~1,500 words + 1 axis figure)
§ 10  Reproducibility                           (~250 words)
§ 11  References (ETCSL + secondary)
§ 12  Appendix: pre-flight concept availability (~500 words)
```

Deep-dive ordering recapitulates the cosmogonic arc: `abzu` (pre-creation) → `zi` (human animation) → `nam` (essence-prefix machinery) → `nam-tar` (fate-determination) → `me` (civilizational distribution).

### Per-concept 8-section template (applied in §4-8)

| Sub-section | Method | ~length |
|---|---|---|
| 1. Anchor reading | ePSD2 gloss + Sumerological context | ~150 words |
| 2. Nearest Sumerian neighbors | `find(english_anchor, space="gemma")` top-10 | ~250 words + table |
| 3. Semantic-field map | 15-20 thematically-adjacent Sumerian terms; pairwise cosine distances; heatmap | ~200 words + figure |
| 4. Dual-view divergence | top-K in Gemma vs. GloVe; agreements high-confidence; disagreements = facet visible to only one space | ~200 words + comparison table |
| 5. Analogy probes | 2-3 targeted `find_analogy` queries testing specific interpretive claims | ~200 words + results |
| 6. English displacement | cosine(Sumerian-projected, English-native) in Gemma and GloVe | ~100 words + single-number callout |
| 7. Source-text grounding | 1-2 ETCSL passages showing the concept in narrative use | ~200 words + quoted passage |
| 8. Interpretive synthesis | draw geometric findings into one hedged cosmogonic claim | ~300 words |

### Manual vs. automated boundary

| Component | Generated | Hand-written |
|---|---|---|
| Abstract, Introduction, Methodology | | ✓ |
| Narrative spine prose | | ✓ |
| UMAP figure (narrative-spine opener) | ✓ | |
| Deep-dive prose | | ✓ |
| Tables (neighbors, analogies, displacement) | ✓ | |
| Heatmap figures | ✓ | |
| ETCSL passages (retrieval) | ✓ | |
| ETCSL passages (contextualization) | | ✓ |
| Synthesis axis figure | ✓ | |
| Synthesis interpretive prose | | ✓ |
| Reproducibility, References | | ✓ |
| Pre-flight appendix | ✓ | |

**Discipline:** prose is written AFTER tables and figures exist. Geometric claims trace to committed data. Contradictions with pre-existing hypothesis get acknowledged in prose rather than silently discarded.

### Pre-flight concept check workflow

Run FIRST, before any prose is written. For each of the 5 candidate concepts:

1. **Vocab check:** is the Sumerian token in the fused 35,508-word vocab?
2. **English anchor check:** is the English seed in Gemma vocab AND GloVe vocab?
3. **Top-5 quality check:** run `find_both(english_seed)`. Flag anchors whose top-5 is dominated by single-letter or clearly-degenerate matches (the `sirara→c` pattern from the 2a audit).
4. **ETCSL passage count:** how many passages contain the token? Flag if zero.

Output: `results/cosmogony_preflight_<YYYY-MM-DD>.json` with per-concept verdicts.

Substitution rules if a concept fails:
- Vocab miss → substitute from pre-approved alternates: `im-a` (clay), `kur` (netherworld), `an`, `ki`.
- Degenerate top-5 → investigate rather than auto-substitute; likely an anchor-quality finding worth flagging.
- Zero ETCSL passages → substitute; no passages means no §7 grounding possible.

Substitutions documented in §12 appendix so readers see exactly what was swapped.

### Analytical tooling contracts

All modules take a `SumerianLookup` instance (already exists in `final_output/sumerian_lookup.py`) plus concept-specific inputs.

```python
# scripts/analysis/semantic_field.py
def compute_pairwise_distances(
    lookup: SumerianLookup,
    sumerian_tokens: list[str],
    space: str = "gemma",
) -> np.ndarray: ...

def render_semantic_field_heatmap(
    distances: np.ndarray,
    tokens: list[str],
    title: str,
    out_path: Path,
) -> None: ...

# scripts/analysis/english_displacement.py
def english_displacement(
    lookup: SumerianLookup,
    sumerian_token: str,
    english_seed: str,
    space: str = "gemma",
) -> dict: ...

# scripts/analysis/etcsl_passage_finder.py
def find_passages(
    sumerian_token: str,
    etcsl_texts: list[dict],
    max_passages: int = 3,
    context_lines: int = 2,
) -> list[dict]: ...

# scripts/analysis/umap_projection.py
def umap_cosmogonic_vocabulary(
    lookup: SumerianLookup,
    tokens: list[str],
    labels: dict[str, str],
    space: str = "gemma",
    out_path: Path = None,
) -> None: ...

# scripts/analysis/preflight_concept_check.py
def preflight_check(
    lookup: SumerianLookup,
    candidate_concepts: list[dict],
    etcsl_texts: list[dict],
) -> dict: ...
```

Two top-level entry points orchestrate:
- `generate_cosmogony_figures.py main()` → all 7 PNGs under `docs/figures/cosmogony/`.
- `generate_cosmogony_tables.py main()` → `docs/cosmogony_tables.json` with every numeric table referenced in the document.

Both are deterministic (fixed seed for UMAP, sorted iteration throughout) so regeneration produces byte-identical output when inputs are unchanged.

### Dependencies

New pip packages beyond current `requirements.txt`:

- **umap-learn** — already planned per `RESEARCH_VISION.md` dependencies list.
- **matplotlib** — already a transitive dep via gensim; pin explicitly.

No other new dependencies. No new model files or corpora.

## Error Handling

| Condition | Behavior |
|---|---|
| Sumerian token not in fused vocab | `KeyError` with message listing the missing token and suggesting pre-flight substitution. |
| English seed not in target vocab | Pre-flight flags it; substitution triggered if fatal. |
| `find_passages` returns zero results | `§7` of that deep dive notes the absence rather than faking a passage. If this is a surprise (passed pre-flight but no passages retrievable post-normalization), it's a real finding worth noting. |
| UMAP fails to converge on small vocabulary (<15 tokens) | Use a lower `n_neighbors` parameter; if still fails, fall back to PCA 2D projection with a note in the figure caption. |
| ETCSL JSON path missing | `FileNotFoundError` pointing to `scripts/01_scrape_etcsl.py`. |
| `cosmogony_tables.json` missing at document-build time | Hard fail; the document consumes its own tooling's output, not hand-written numbers. |

## Testing Strategy

New `tests/analysis/` directory. Each analysis module gets a matching test file using tiny synthetic inputs.

### `test_semantic_field.py`
- `test_pairwise_distances_is_symmetric`
- `test_pairwise_distances_diagonal_is_zero`
- `test_heatmap_renders_png_to_path`

### `test_english_displacement.py`
- `test_displacement_computes_cosine_in_gemma_space`
- `test_displacement_skips_oov_english_seed`

### `test_etcsl_passage_finder.py`
- `test_returns_passages_with_token`
- `test_context_lines_captured`
- `test_zero_passages_when_token_absent`

### `test_umap_projection.py`
- `test_umap_runs_on_synthetic_tokens`
- `test_falls_back_to_pca_when_vocabulary_too_small`
- `test_deterministic_with_fixed_seed`

### `test_preflight_concept_check.py`
- `test_all_concepts_pass_when_vocab_complete`
- `test_flags_degenerate_top5`
- `test_flags_zero_etcsl_passages`
- `test_report_json_schema_stable`

All tests run in < 5 seconds total. No dependence on real alignment artifacts — synthetic `SumerianLookup` stubs used throughout.

## Reproducibility

- Every numeric claim in prose → row in `cosmogony_tables.json` → deterministic output of `generate_cosmogony_tables.py`.
- Every figure → PNG → deterministic output of `generate_cosmogony_figures.py`.
- Pre-flight report → dated JSON under `results/`.
- Document's §10 names the git commit whose alignment artifacts produced the numbers. If anyone reruns the alignment pipeline (new anchors, new FastText), they diff the regenerated tables against the pinned version.

## Operational notes

**Success path:** Developer runs pre-flight, reviews report, accepts or substitutes concepts, runs table and figure generators, writes prose section-by-section referencing the generated artifacts, commits in logical groups (tooling → pre-flight report → tables → figures → prose), journals the completion.

**Failure paths:**
- Pre-flight fails on 2+ concepts: the substitution slate may be insufficient; escalate with an additional brainstorm before writing.
- Numeric finding contradicts a pre-existing interpretive hypothesis: acknowledge in prose. This is part of what keeps the document honest.
- UMAP figure looks uninterpretable: fall back to PCA, document the fallback in §3 figure caption. Not a showstopper.

## Follow-up work (out of this spec)

- Comparative analysis (Egyptian via Heiroglyphy) — separate repo.
- Phase 2 Riemannian geometry — deferred to future spec.
- Steering / interp bridge — separate research track.
- Anchor semantic-quality pass — could be triggered by findings (e.g., if deep dives surface consistent anchor-noise patterns).
- Additional cosmogonic domains: fate/meaning/self (Phase 1's other two domains) — natural sequels if the first document lands well.

The document's §9 synthesis section explicitly flags which future directions the findings support or complicate, giving concrete hooks for follow-up workstreams.
