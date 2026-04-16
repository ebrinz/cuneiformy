# EmbeddingGemma as English Target Space (Phase A)

**Date:** 2026-04-16
**Status:** Approved (brainstorming), pending writing-plans
**Branch:** feat/sumerian-alignment (current)

## Summary

Run a direct benchmark comparing GloVe 300d vs `google/embeddinggemma-300m` 768d as the English target space for Ridge alignment of Sumerian FastText vectors. Phase A is a gated experiment — if EmbeddingGemma improves top-1 accuracy by ≥3pp over the current 11.24% baseline *and* a qualitative read of concept-cluster neighbors is more coherent, phase B (downstream pipeline swap for the concept-cluster work in `NEAR_TERM_STRATEGY.md`) proceeds as a separate spec. If it doesn't, we've learned the alignment ceiling is on the Sumerian side, not the English target side, which redirects the research plan.

## Motivation

The current pipeline aligns fused 1536d Sumerian vectors (FastText 768d + zero-padding) into GloVe 300d English via Ridge regression at 11.24% top-1. GloVe is static, 2014-era Common Crawl, and caps at 300d — part of the accuracy ceiling is plausibly the expressiveness of the target space, not the Sumerian side. EmbeddingGemma is a 2025-era contextual encoder trained on modern multilingual data at 768d. It matches the Sumerian input dimension and is a natural drop-in for the target. This experiment isolates "does a richer English target help?" as a yes/no question before investing in the much larger move of contextualizing the Sumerian side (fine-tuning Gemma on Sumerian transliteration).

## Scope

### In scope
- New alignment script `09b_align_gemma.py` using EmbeddingGemma as the target space
- One-shot precompute of EmbeddingGemma vectors for GloVe's 400k vocabulary, via WordNet glosses
- Qualitative concept-cluster comparison script (human-read report, no automated pass/fail)
- Reuse of existing fused 1536d Sumerian vectors, existing anchor set, existing Ridge hyperparameters

### Out of scope
- Fine-tuning Gemma on Sumerian transliteration (phase 2 of the broader research plan)
- Contextualizing the Sumerian side
- Changes to `09_align_and_evaluate.py` (baseline stays reproducible)
- Changes to the fusion step, anchor extraction, or corpus preprocessing
- Updating `final_output/sumerian_aligned_vectors.npz` or `sumerian_lookup.py` (phase B territory)
- Multiple Gemma variants, matryoshka truncation experiments, alternative ridge solvers

## Success Criteria (Gate for Phase B)

Phase A **passes** and phase B is greenlit if both:

1. **Quantitative:** Top-1 accuracy on the held-out 20% anchor test set improves by **≥3pp** over the 11.24% GloVe baseline (i.e., ≥14.24%).
2. **Qualitative:** For ~20 concept-cluster seed words drawn from `NEAR_TERM_STRATEGY.md` (creation/fate/self domains), the top-10 Sumerian-projected nearest neighbors in the Gemma-aligned space are judged (by the project owner, reading the comparison report) more semantically coherent than in the GloVe-aligned space.

Phase A **fails** (and we do not proceed to phase B) if either criterion is not met. A failing result redirects the research agenda — it implies the Sumerian-side embedding is the bottleneck, making Gemma fine-tuning on Sumerian corpus the next move rather than downstream pipeline work.

## Design

### Architecture

Three new scripts, one cached data artifact, two new output artifacts. No modifications to existing pipeline files.

```
scripts/
  embed_english_gemma.py       # one-shot: GloVe vocab → Gemma vectors via WordNet glosses
  09b_align_gemma.py           # alignment + quantitative eval (mirrors 09_align_and_evaluate.py)
  evaluate_concept_clusters.py # qualitative top-10 comparison report

models/
  english_gemma_768d.npz       # cached (vocab, vectors) for 400k words — built once
  ridge_weights_gemma.npz      # ridge coef/intercept for Sumerian → Gemma

results/
  alignment_results_gemma.json # top-k metrics + provenance (mirrors alignment_results.json)
  concept_clusters_comparison.md # human-read side-by-side GloVe vs Gemma top-10
```

### Data Flow

```
[ONE-SHOT PRECOMPUTE — embed_english_gemma.py]

  GloVe vocab (400k words, from data/processed/glove.6B.300d.txt)
       │
       ▼
  WordNet synset lookup ──► "{word}: {first synset definition}"  (hit: ~85%)
       │                     or
       ▼                    "{word}"                              (miss: ~15%)
  EmbeddingGemma encoder (document prompt template, batch=64)
       │
       ▼
  models/english_gemma_768d.npz   { vocab: (N,) str, vectors: (N, 768) f32 }


[PHASE A ALIGNMENT — 09b_align_gemma.py]

  models/fused_embeddings_1536d.npz ──┐
  models/english_gemma_768d.npz ──────┼──► build_training_data
  data/processed/english_anchors.json ┘         │
                                                ▼
                                       train/test split (80/20, random_state=42)
                                                │
                                                ▼
                                         Ridge(alpha=100)
                                                │
                                    ┌───────────┴────────────┐
                                    ▼                        ▼
                        evaluate_alignment (top-1/5/10)   ridge_weights_gemma.npz
                                    │
                                    ▼
                        results/alignment_results_gemma.json


[QUALITATIVE GATE — evaluate_concept_clusters.py]

  SumerianLookup (GloVe, existing final_output/sumerian_lookup.py)
       │
       │  for each of ~20 seed words from NEAR_TERM_STRATEGY.md:
       │    English seed → Sumerian nearest neighbors (k=10)
       │    → English re-projection of those Sumerian words
       ▼
  _GemmaLookup (in-memory, built from ridge_weights_gemma.npz + english_gemma_768d.npz)
       │
       │  same procedure in Gemma-aligned space
       ▼
  results/concept_clusters_comparison.md  (side-by-side columns per seed)
```

### Components

**`scripts/embed_english_gemma.py`** — one-shot English vocab precompute.
- Reads GloVe vocabulary from `data/processed/glove.6B.300d.txt` (first column of each line only; do not load the 300d vectors).
- For each word, looks up the first synset via `nltk.corpus.wordnet.synsets(word)[0].definition()`. Hit → format as `"{word}: {definition}"`. Miss → use bare `"{word}"`.
- Encodes with `google/embeddinggemma-300m` using the model's document prompt template (per model card). Batch size 64 by default. Device autodetect (MPS > CUDA > CPU).
- Writes `models/english_gemma_768d.npz` with `vocab` and `vectors` arrays.
- Logs: gloss hit rate, per-batch progress, encoding failures.
- **Idempotent:** if output file exists and its vocab matches the GloVe vocab exactly, skip encoding and exit 0.
- Pins the Gemma model by commit hash via a module-level constant.

**`scripts/09b_align_gemma.py`** — alignment + quantitative evaluation.
- Reuses `build_training_data`, `train_ridge`, `evaluate_alignment` from `09_align_and_evaluate.py`. Preferred path: import directly (the scripts package already works for other 0N-prefixed modules; note that `scripts/align_09.py` appears to be an alternate filename without the digit-prefix issue). If importing requires modifying the baseline script or its imports, fall back to verbatim copy-paste of the three helpers into `09b_align_gemma.py` — duplication is acceptable here to preserve the "baseline stays untouched" invariant from scope.
- Loads `fused_embeddings_1536d.npz` as X, `english_gemma_768d.npz` as Y source, `english_anchors.json` as pair list.
- Same hyperparameters as GloVe baseline: `test_size=0.2`, `random_state=42`, `alpha=100`.
- Saves ridge weights to `models/ridge_weights_gemma.npz`.
- Writes `results/alignment_results_gemma.json` with accuracy metrics, config (including Gemma model hash, WordNet version, miss rate), and a comparison block showing delta vs GloVe baseline.
- Prints to stdout: `Top-1: 11.24% → X.XX% (+Y.YYpp vs GloVe)`.
- **Assertion:** fails loudly if `Y.shape[1] != 768` before ridge fit (guards against a stale cache).

**`scripts/evaluate_concept_clusters.py`** — qualitative comparison.
- Seed word list as a module-level constant, pulled from `NEAR_TERM_STRATEGY.md`:
  - Creation: `create`, `begin`, `birth`, `origin`, `emerge`, `form`, `separate`
  - Fate/meaning: `fate`, `destiny`, `purpose`, `decree`, `name`, `order`
  - Self/soul: `self`, `soul`, `spirit`, `mind`, `heart`, `breath`, `shadow`
- For each seed: execute the "reverse query" pattern from strategy doc — English seed → top-10 Sumerian nearest neighbors → for each of those Sumerian words, top-5 English nearest neighbors **in the same space being evaluated** (i.e., when running the GloVe condition, English re-projection uses GloVe; when running the Gemma condition, English re-projection uses the cached Gemma vectors). The two conditions must be symmetric — no cross-space lookups.
- Run procedure twice: once against existing `SumerianLookup` (GloVe-backed, uses GloVe for both Sumerian projection and English nearest-neighbor lookup), once against a minimal `_GemmaLookup` helper class defined inline in this script that wraps `ridge_weights_gemma.npz` + `english_gemma_768d.npz` with the same API shape (uses Gemma vectors for both sides).
- Output `results/concept_clusters_comparison.md` with one section per seed, two columns side by side (GloVe | Gemma), and a short header noting that judgment is human-read.
- No automated pass/fail. The report is the deliverable.

### Error Handling

- **Gemma model download/auth failure.** Propagate the HuggingFace error. No retry logic.
- **`nltk` WordNet not installed.** Detect early in `embed_english_gemma.py`; print `nltk.download('wordnet')` instruction; exit 1. No auto-download.
- **WordNet synset miss.** Log and fall back to bare-word encoding. Not an error. Miss rate reported in results JSON.
- **Encoder OOM or device error.** Catch per-batch, halve batch size once, retry. If it fails again, propagate.
- **Anchor vocabulary mismatch.** Existing `build_training_data` skips anchors missing from either vocab. No change needed.
- **Dimension mismatch at ridge fit.** Explicit assertion in `09b_align_gemma.py` before `model.fit`.

### Testing

- `tests/test_embed_english_gemma.py` — unit tests for gloss formatting (word + definition → `"{word}: {definition}"`) and WordNet lookup wrapper (known hit, known miss). Does not test the encoder.
- `tests/test_09b_align_gemma.py` — shape-contract test with synthetic 10-anchor, 50d input verifying `build_training_data` / `train_ridge` / `evaluate_alignment` work when the target space is 768d.
- **No test for `evaluate_concept_clusters.py`.** Reporting script with human-judged output.
- **No live-model integration test.** The successful production of `alignment_results_gemma.json` is the integration test.

### Reproducibility

- Pin `google/embeddinggemma-300m` by git commit hash in a module constant.
- Record in `alignment_results_gemma.json`: Gemma model hash, prompt template string, WordNet data version (`nltk.data.find('corpora/wordnet').path` inspection), gloss hit rate, bare-word fallback count, all ridge hyperparameters, anchor counts, dimensions.

## Non-Decisions (Explicit)

Items considered and deliberately deferred out of phase A:

- **Matryoshka truncation (512/256/128d).** Use full 768d. Truncation is an optimization for serving, not for this benchmark.
- **Alternative gloss sources (Wiktionary, ConceptNet, LLM-generated).** WordNet's ~85% coverage of GloVe's 400k is sufficient; misses fall back cleanly.
- **Contextualizing the Sumerian side.** Explicitly phase 2. Requires fine-tuning and is a separate spec.
- **Re-evaluating against the `SumerianLookup` production API.** Phase A does not touch `final_output/`. If phase A passes, phase B brainstorm decides how that API changes.

## Risks

- **Gemma may not beat GloVe.** This is the expected-value of the experiment. A null result is informative and redirects the research plan toward Sumerian-side work.
- **Single-word gloss approach is a heuristic.** The first WordNet synset isn't always the most common sense. This introduces noise into the English target space but applies uniformly across anchors and candidates, so it shouldn't systematically bias the comparison.
- **Compute budget.** ~400k encoder calls at batch 64. On M-series MPS: ~10-30 min. On CPU: 1-2 hours. Acceptable for a one-shot cache.
- **Anchor set quality ceiling.** If the 97k anchor set has a high noise floor, no target space swap will materially move top-1. Phase A cannot distinguish "Gemma didn't help" from "anchors are the problem." A null result should prompt a separate anchor-quality audit before declaring the Sumerian-side is the bottleneck.

## References

- `docs/NEAR_TERM_STRATEGY.md` — source of concept-cluster seed words and the downstream use case
- `docs/RESEARCH_VISION.md` — broader framing for why target-space quality matters
- `scripts/09_align_and_evaluate.py` — baseline pipeline being compared against
- `scripts/08_fuse_embeddings.py` — confirms fused 1536d input is GloVe-free (FastText + zero-pad)
- EmbeddingGemma model card: `google/embeddinggemma-300m` on HuggingFace Hub
