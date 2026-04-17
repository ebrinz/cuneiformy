# Phase B: Dual-view Downstream Pipeline (Whitened-Gemma Primary, GloVe Secondary)

**Date:** 2026-04-16
**Status:** Approved (brainstorming), pending writing-plans
**Branch:** `feat/sumerian-alignment` (current, extended)
**Follows:** `docs/superpowers/specs/2026-04-16-gemma-embed-alignment-design.md` (Phase A)
**Journal:** `docs/EXPERIMENT_JOURNAL.md`

## Summary

Extend `final_output/` and `SumerianLookup` so downstream analysis can operate in *both* the whitened-EmbeddingGemma 768d manifold (primary) and the existing GloVe 300d manifold (secondary). After Phase A and its retries landed whitened-Gemma at 19.85% top-1 (+2.54pp vs GloVe) with qualitatively different/complementary concept clusters, the research case for moving the downstream substrate to Gemma is clear even though the formal +3pp gate was missed by 0.46pp. Keeping GloVe accessible as a parallel lens preserves its strengths on kinship/concrete concepts where it outperforms Gemma qualitatively.

## Motivation

The research aim (per `docs/RESEARCH_VISION.md`) is geometric — curvature, manifold shape, cross-civilizational comparison — not translation accuracy. For that, the target space's dimensionality (768d > 300d), multilinguality (Gemma supports 100+ languages natively, which matters when Akkadian, Egyptian, Classical Greek enter the pipeline), and forward-compatibility with contextual Sumerian (future Gemma-generative fine-tune on ETCSL/CDLI/ORACC) all favor Gemma. The +2.54pp alignment win is a secondary benefit; the primary gain is that the downstream work now lives in the space where the research actually wants to happen.

Keeping GloVe as a second view is cheap (artifacts already exist) and operationally useful — the phase A qualitative report showed Gemma and GloVe surface different facets of the same Sumerian concepts.

## Scope

### In scope
- New artifact `final_output/sumerian_aligned_gemma_vectors.npz` — 35,508 Sumerian words projected into whitened-Gemma 768d via the ridge weights from `models/ridge_weights_gemma_whitened.npz`.
- Existing `final_output/sumerian_aligned_vectors.npz` (GloVe 300d) retained unchanged.
- Shared `final_output/sumerian_aligned_vocab.pkl` — same 35,508 Sumerian words in the same order across both spaces.
- Updated `final_output/sumerian_lookup.py` — dual-view class with `space="gemma"|"glove"` parameter, `"gemma"` default.
- Updated `scripts/10_export_production.py` — produces both spaces' aligned vectors in one run; writes consolidated `metadata.json` with both spaces' provenance.
- New `scripts/validate_phase_b.py` — sanity-checks the new exports with live `find()` calls and regenerates the concept-cluster comparison report as a regression check.
- Updated `tests/test_10_export.py` — covers dual-view API, routing, error paths, export roundtrip.

### Out of scope
- Geometric analysis described in `NEAR_TERM_STRATEGY.md` Phase 1 (distance matrices, centroid displacement, cluster eigenvalue decomposition, UMAP projections, narrative extraction). Becomes its own brainstorm once Phase B ships.
- Runtime EmbeddingGemma encoding for out-of-vocab English queries. Class operates on the fixed 400k whitened English cache.
- Anchor quality audit (Workstream 2a, separate brainstorm).
- Any Sumerian FastText changes or Gemma fine-tuning.
- New `SumerianLookup` methods beyond the existing `find` / `find_analogy` / `find_blend`, plus one convenience `find_both`.

### Deliverables produced
- One new vector file, one updated lookup class, one new validation script, extended tests, updated metadata.
- Regression check: running the validation script regenerates `results/concept_clusters_comparison_whitened.md` identical in shape to the one produced during Phase A retry #2.

## Success Criteria

- `SumerianLookup(...).find("king", space="gemma")` returns a non-empty top-10 Sumerian list.
- `SumerianLookup(...).find("king", space="glove")` returns a non-empty top-10 Sumerian list.
- `SumerianLookup(...).find_both("fate")` returns a dict with `"gemma"` and `"glove"` keys, both non-empty.
- Existing test suite (`pytest tests/`, minus integration) still passes.
- New dual-view tests pass.
- `scripts/validate_phase_b.py` exits 0.
- `final_output/metadata.json` schema_version is 2, includes accuracy and config for both spaces, alpha=100 (not the stale 0.001 in the current file).

## Design

### Architecture

```
final_output/
  sumerian_aligned_vectors.npz          # existing, GloVe 300d fp16 (unchanged shape)
  sumerian_aligned_gemma_vectors.npz    # NEW — whitened Gemma 768d fp16
  sumerian_aligned_vocab.pkl            # existing, shared across both spaces
  sumerian_lookup.py                    # UPDATED — dual-view with space="gemma"|"glove"
  metadata.json                         # UPDATED — schema_version=2, both spaces, correct alpha

scripts/
  10_export_production.py               # UPDATED — also projects + saves Gemma-space
  validate_phase_b.py                   # NEW — sanity + regression check

tests/
  test_10_export.py                     # UPDATED — dual-view coverage
```

Files NOT touched: `scripts/09_align_and_evaluate.py`, `scripts/09b_align_gemma.py`, `scripts/embed_english_gemma.py`, `scripts/whiten_gemma.py`, `scripts/evaluate_concept_clusters.py`, `scripts/align_09.py`, `scripts/align_09b.py`, and all upstream scraping / tokenization / fusion scripts.

### Data Flow (Export)

```
models/fused_embeddings_1536d.npz ─┐
                                    │
models/ridge_weights.npz ───────────┼─► project ─► final_output/sumerian_aligned_vectors.npz (GloVe 300d)
                                    │
models/ridge_weights_gemma_         │
      whitened.npz ─────────────────┴─► project ─► final_output/sumerian_aligned_gemma_vectors.npz (Gemma 768d)

Vocab passthrough (shared) ─────────► final_output/sumerian_aligned_vocab.pkl
Metadata aggregation ───────────────► final_output/metadata.json (schema_version=2)
```

The Gemma-space ridge weights were trained against *already-whitened* English targets during Phase A retry #2, so the projection `sum_fused @ coef.T + intercept` directly lands in the whitened manifold. No additional transform at export time.

### Components

**`scripts/10_export_production.py` (updated)**

Single-run export of both spaces. Loads:
- `models/fused_embeddings_1536d.npz` — fused Sumerian (once).
- `models/ridge_weights.npz` — GloVe alignment ridge.
- `models/ridge_weights_gemma_whitened.npz` — whitened-Gemma alignment ridge.
- `results/alignment_results.json` and `results/alignment_results_gemma_whitened.json` — for accuracy numbers in metadata.

Projects Sumerian into each target space, saves as fp16 compressed npz in `final_output/`. Writes metadata with:
- `schema_version: 2`
- `spaces.gemma`: dim, dtype, vocab_size, ridge_alpha, ridge_source_path, accuracy (top-1/5/10), gloss_hit_rate, gemma_model name, whitening_transform_path
- `spaces.glove`: dim, dtype, vocab_size, ridge_alpha, ridge_source_path, accuracy
- `shared.vocab_size`: 35508
- `shared.sumerian_fused_dim`: 1536
- `shared.anchor_stats`: train/test sizes, total anchors, valid_anchors
- `shared.random_state`: 42

**`final_output/sumerian_lookup.py` (rewritten, dual-view)**

Single class `SumerianLookup`. Constructor loads both aligned Sumerian arrays (float32-upcast from fp16), the shared vocab, and both English vocab+vector sources. Pre-normalizes all four arrays (Sumerian×2, English×2) for cosine.

```python
class SumerianLookup:
    def __init__(
        self,
        gemma_vectors_path: str,
        glove_vectors_path: str,
        vocab_path: str,
        gemma_english_path: str,
        glove_english_vectors: np.ndarray,
        glove_english_vocab: list[str],
    ): ...

    def find(self, english_word: str, top_k: int = 10, space: str = "gemma") -> list[tuple[str, float]]: ...
    def find_both(self, english_word: str, top_k: int = 10) -> dict[str, list[tuple[str, float]]]: ...
    def find_analogy(self, a: str, b: str, c: str, top_k: int = 10, space: str = "gemma") -> list[tuple[str, float]]: ...
    def find_blend(self, weights: dict[str, float], top_k: int = 10, space: str = "gemma") -> list[tuple[str, float]]: ...
```

Internal helpers:
- `_english_vector(word, space)` → normalized float32 vector or `None` for OOV.
- Space routing table: `_spaces = {"gemma": (sum_gemma_norm, eng_gemma_vocab_map, eng_gemma_norm), "glove": (...)}`.

**`scripts/validate_phase_b.py` (new)**

Reads the newly-exported artifacts, builds a `SumerianLookup`, runs:
- `find("king")`, `find("fate")`, `find("soul")` in each space → prints top-5 per space side-by-side.
- `find_both("name")` → prints both keys' top-3.
- `find_analogy("king", "queen", "father")` in both spaces → prints top-3 per space.
- Shell-calls `python scripts/evaluate_concept_clusters.py --gemma-mode whitened` and asserts the output file `results/concept_clusters_comparison_whitened.md` exists and contains `"## Domain: creation"`, `"## Domain: fate_meaning"`, `"## Domain: self_soul"` headers.

Exits 0 on success, 1 with a specific error message on any failure.

### Error Handling

- Missing `sumerian_aligned_gemma_vectors.npz` at construction → `FileNotFoundError` with message naming `scripts/10_export_production.py`.
- Vocab length ≠ Sumerian vector count → `AssertionError` before any lookup is attempted.
- Gemma English cache dim ≠ 768 → `AssertionError` pointing to `scripts/whiten_gemma.py`.
- GloVe English vectors dim ≠ 300 → `AssertionError`.
- `space` not in `{"gemma", "glove"}` → `ValueError("space must be 'gemma' or 'glove', got {space!r}")`.
- OOV English word → empty list (not an error; matches current GloVe-only behavior).
- Empty blend (no words resolved) → empty list, no div-by-zero.

### Testing

All in `tests/test_10_export.py`:

- `test_sumerian_lookup_find_gemma_returns_top_k`: synthetic 5-word Sumerian, 3-word English, top-1 is the deliberately-aligned pair.
- `test_sumerian_lookup_find_glove_returns_top_k`: same construction via GloVe path.
- `test_sumerian_lookup_find_both_has_both_keys`: asserts `set(result.keys()) == {"gemma", "glove"}` and both lists non-empty.
- `test_sumerian_lookup_unknown_space_raises`: `pytest.raises(ValueError)` on `space="bert"`.
- `test_sumerian_lookup_oov_returns_empty_list`: missing English word → `[]` for both spaces.
- `test_sumerian_lookup_analogy_routes_by_space`: analogy in gemma vs glove returns different top-k lists given deliberately-different underlying vectors.
- `test_sumerian_lookup_blend_empty_weights_returns_empty`: no resolvable words → `[]`.
- `test_export_roundtrip`: tiny fake fused + two fake ridges → run export function → load via SumerianLookup → one find succeeds with expected top-1.

No live-model tests. No integration with real anchors. Fixtures use small arrays constructed in the test.

### Reproducibility

`final_output/metadata.json` schema (v2):

```json
{
  "schema_version": 2,
  "methodology": "Cuneiformy dual-view (Sumerian 1536d -> whitened-Gemma 768d primary, GloVe 300d secondary)",
  "shared": {
    "vocab_size": 35508,
    "sumerian_fused_dim": 1536,
    "random_state": 42,
    "train_size": 1572,
    "test_size": 393,
    "valid_anchors": 1965,
    "total_anchors": 13886
  },
  "spaces": {
    "gemma": {
      "dim": 768,
      "dtype": "float16",
      "ridge_alpha": 100,
      "ridge_source": "models/ridge_weights_gemma_whitened.npz",
      "target_source": "models/english_gemma_whitened_768d.npz",
      "whitening_transform": "models/gemma_whitening_transform.npz",
      "encoder_model": "google/embeddinggemma-300m",
      "encoder_prompt": "Retrieval-document",
      "gloss_source": "WordNet first synset",
      "gloss_hit_rate_pct": 21.4,
      "accuracy": {"top1": 19.85, "top5": 23.66, "top10": 26.21}
    },
    "glove": {
      "dim": 300,
      "dtype": "float16",
      "ridge_alpha": 100,
      "ridge_source": "models/ridge_weights.npz",
      "target_source": "data/processed/glove.6B.300d.txt",
      "accuracy": {"top1": 17.30, "top5": 22.90, "top10": 25.19}
    }
  }
}
```

## Non-Decisions (Explicit)

- **Runtime Gemma encoding for OOV words.** Deferred. Would add ~600MB memory and a 1-2s first-call model load. Query words outside the 400k vocab return `[]`.
- **Dropping GloVe.** Kept as secondary view. Cost is a ~30MB extra npz file and ~40 lines of class code; benefit is complementary qualitative coverage.
- **Copying the 1.1 GB `english_gemma_whitened_768d.npz` into `final_output/`.** Not copying. `SumerianLookup` takes a path; callers point at `models/`.
- **Phase 1 geometric analysis from `NEAR_TERM_STRATEGY.md`.** Defer to a separate brainstorm once Phase B is shipped.

## Risks

- **Consumers outside the repo might depend on the old single-space `SumerianLookup` API.** Mitigation: this is a research project, there are no known external consumers, and `test_10_export.py` is the only in-repo consumer. `schema_version` in metadata signals the break.
- **The whitening transform is tied to one specific encoded-gloss cache.** If the English cache is regenerated with different gloss sources, the whitening transform and therefore the ridge weights become invalid. Mitigation: metadata records both file paths; validation script fails loudly if dims disagree.
- **Float16 compression introduces precision loss.** Existing GloVe-space export already uses fp16 and downstream queries work fine; same assumption here. Class upcasts to float32 at load time before cosine normalization.

## References

- `docs/superpowers/specs/2026-04-16-gemma-embed-alignment-design.md` — Phase A spec (the alignment experiment this phase consumes)
- `docs/superpowers/plans/2026-04-16-gemma-embed-alignment.md` — Phase A plan
- `docs/EXPERIMENT_JOURNAL.md` — Phase A retries including the whitening breakthrough
- `docs/NEAR_TERM_STRATEGY.md` — downstream geometric analysis plan (next brainstorm)
- `docs/RESEARCH_VISION.md` — research thesis motivating the substrate choice
