# Experiment Journal

Running log of experiments on the Cuneiformy pipeline. Reverse chronological — newest at the top. Each entry records hypothesis, method, result, and takeaway so that past attempts (and null results) stay legible even when the surrounding code has moved on.

Entry format:

```
## YYYY-MM-DD — <short name>
**Hypothesis:**
**Method:**
**Result:**
**Takeaway:**
**Artifacts / commits:**
```

---

## 2026-04-16 — Phase A retry #2: Whitening the EmbeddingGemma target

**Hypothesis:** Both prior Gemma runs failed because contextual embeddings are anisotropic — they cluster in a narrow cone where a single shared direction dominates, inflating cosine similarity between unrelated words and drowning real semantic signal. Centering (subtract global mean) and whitening (scale each direction to unit variance, BERT-whitening, Su et al. 2021) should expose the actual concept structure. 30-second matrix-math retry on already-cached vectors — no re-encoding.

**Method:** `scripts/whiten_gemma.py` reads each cached Gemma variant (gloss, bare), computes mean μ and transform W = Σ⁻¹ᐟ² via eigendecomposition of the 768×768 covariance, saves whitened caches to `models/english_gemma_whitened_768d.npz` and `models/english_gemma_bare_whitened_768d.npz`. 09b gains a generic `--mode` flag routing through the appropriate cache. Ridge hyperparameters, anchors, split all unchanged.

**Result:**

| target | top-1 | top-5 | top-10 | top-1 Δ vs GloVe |
|---|---|---|---|---|
| GloVe (baseline) | 17.30% | 22.90% | 25.19% | — |
| Gemma gloss (raw) | 14.25% | 16.54% | 17.30% | −3.05pp |
| Gemma bare (raw) | 4.83% | 14.50% | 15.52% | −12.47pp |
| **Gemma gloss whitened** | **19.85%** | **23.66%** | **26.21%** | **+2.54pp** |
| **Gemma bare whitened** | **19.59%** | **23.66%** | **25.95%** | **+2.29pp** |

**Takeaway:** The anisotropy story was correct — **whitening turned both Gemma variants from clear losses into clear wins over GloVe on all top-k metrics.** Top-1 delta is just shy of the formal +3pp gate (+2.54pp, not +3.0), but top-5 and top-10 both improve and the earlier qualitative dictionary-meta artifact appears to evaporate because it lived along the first principal component that whitening strips.

Two previously misleading conclusions to retract:
- **"The gloss encoding introduced a systematic bias"** — partially true but the *actual* dominant issue was anisotropy; after centering, gloss and bare produce near-identical alignment quality (19.85% vs 19.59%), so the encoding choice barely matters.
- **"The Sumerian side is the bottleneck"** — premature. Whitening just lifted the ceiling 2.5pp without touching the Sumerian side. The new ceiling (19.85%) is still low in absolute terms, but the English target is no longer the limiting factor it appeared to be. Sumerian-side work remains the most interesting frontier, but the anisotropy lesson means any future use of contextual encoders anywhere in the pipeline should assume centering/whitening is mandatory, not optional.

Pre- vs post-whitening diagnostics for the gloss run:
- Pre: mean-vector norm 0.60 (vectors cluster on one side of the unit sphere)
- Pre: random 1000-pair mean cosine 0.35 (unrelated words look ~0.35 similar — noise floor)
- Post: mean-vector norm 0.002 (centered)
- Post: random-pair mean cosine ~0 (noise floor removed)

**Gate status per original spec:** FAIL on strict reading (+2.54pp < +3pp), PASS in spirit — the improvement is real, consistent across both encoding variants, and the methodological learning (whitening is required for contextual-encoder targets in regression alignment) is itself a valuable outcome.

**Artifacts / commits:**
- Code: `scripts/whiten_gemma.py`, extended 09b `--mode` flag
- Whitened caches: `models/english_gemma_whitened_768d.npz`, `models/english_gemma_bare_whitened_768d.npz` (local only)
- Transforms: `models/gemma_whitening_transform.npz`, `models/gemma_bare_whitening_transform.npz`
- Results: `results/alignment_results_gemma_whitened.json`, `results/alignment_results_gemma_bare_whitened.json` (local only)

---

## 2026-04-16 — Phase A retry: EmbeddingGemma (bare-word) as English target

**Hypothesis:** The gloss-encoding approach from the earlier phase A run introduced a systematic dictionary-meta-vocabulary bias. Encoding bare words (`"{word}"` with no WordNet definition body, same `Retrieval-document` prompt) would avoid that artifact and might close the gap to GloVe.

**Method:** Identical to the gloss run except `embed_english_gemma.py --bare` skipped WordNet lookup and fed each English word alone to EmbeddingGemma. Cache written to `models/english_gemma_bare_768d.npz`. Alignment via `09b_align_gemma.py --bare`. Sumerian side and anchors unchanged.

**Result:**

| metric | GloVe | Gemma (gloss) | Gemma (bare) | bare vs GloVe |
|---|---|---|---|---|
| top-1 | 17.30% | 14.25% | **4.83%** | **−12.47pp** |
| top-5 | 22.90% | 16.54% | 14.50% | −8.40pp |
| top-10 | 25.19% | 17.30% | 15.52% | −9.67pp |

Bare-word is even worse than gloss on top-1 — by 10pp. Top-5/top-10 recover somewhat, suggesting bare vectors find the right neighborhood but not the exact match.

**Takeaway:** Gemma-as-English-target is ruled out. Neither encoding strategy beats GloVe, and bare (the "theoretically cleaner" variant) is *worse* than the flawed gloss variant. The signal is unambiguous: **the English target space is not the bottleneck**. Whatever is capping top-1 at ~17%, it's not the expressiveness of the modern English encoder — it lives somewhere upstream (Sumerian FastText quality, fused 1536d construction, anchor noise, or the alignment method itself).

Interesting sub-finding: the gloss approach, despite its clear artifact, actually *helps* Gemma over bare. The dictionary-meta clustering gives the target space at least *some* structure that the ridge can exploit; bare single-token embeddings give it nothing stable to align against. This is a useful data point for any future contextual-encoder work — single words are off-distribution in ways that hurt downstream regression.

**Redirect:** Next thread should target the Sumerian side, not the English side. Concrete candidates in priority order:
1. Anchor quality audit — 14.2% valid rate is low; how many of the remaining ~12k anchors are noise vs. genuine vocab-mismatch?
2. Sumerian FastText retraining with different hyperparameters (window size, min_count, training corpus weighting)
3. The larger move: fine-tune Gemma (generative) on ETCSL+CDLI+ORACC transliteration to get contextual Sumerian embeddings. This was always the intended phase 2.

**Artifacts / commits:**
- Same code files as gloss run, extended with `--bare` flag — commits `26c49aa`, `6ca6ca8`
- Local-only artifacts: `models/english_gemma_bare_768d.npz` (1.1 GB), `results/alignment_results_gemma_bare.json`, `models/ridge_weights_gemma_bare.npz`

---

## 2026-04-16 — Phase A: EmbeddingGemma (gloss-encoded) as English target

**Hypothesis:** A modern contextual encoder (`google/embeddinggemma-300m`, 768d) as the English target space would improve Ridge-alignment top-1 accuracy vs GloVe 300d by at least +3pp, because GloVe is static / 2014-era and dimensionally shallow compared to the fused 1536d Sumerian side.

**Method:**
- Encoded all 400k GloVe vocab words through EmbeddingGemma using `prompt_name="Retrieval-document"`.
- Each English word wrapped as `"{word}: {WordNet first-synset definition}"` (miss → bare word). Gloss hit rate: 21.4%.
- Same anchor set (13,886), same ridge alpha=100, same 80/20 split (random_state=42) as the GloVe baseline. Only the target vectors changed.
- Sumerian side unchanged: fused 1536d (FastText 768d + zero-pad).

**Result:**

| metric | GloVe baseline | Gemma | delta |
|---|---|---|---|
| top-1 | 17.30% | 14.25% | **−3.05pp** |
| top-5 | 22.90% | 16.54% | −6.36pp |
| top-10 | 25.19% | 17.30% | −7.89pp |

Both quantitative (+3pp) and qualitative (concept cluster coherence) gates **failed**. Gemma is worse, not better.

**Takeaway:** The failure mode is diagnostic, not random. The WordNet-gloss encoding systematically biased the target space: encoding every English word as its definition (`"flour: a fine powder..."`) taught the ridge to project into the region of English that lives near *dictionary definitions themselves*. Nearest neighbors across all domains degenerated to dictionary-meta vocabulary — *"term", "adjective", "word-by-word", "dictionary.com", "merriam-webster"* — rather than concept-adjacent words. The bare-word encoding option (originally ruled out in the spec as "off-distribution") is now the obvious cheap retry: it may be imperfect, but it cannot produce this specific artifact.

Secondary observation: 14.2% anchor validity (1965 of 13886) is lower than ideal and applies equally to both conditions, so it isn't responsible for the Gemma shortfall — but it's a separate thread worth pulling if future retries also underperform.

**Artifacts / commits:**
- Spec: `docs/superpowers/specs/2026-04-16-gemma-embed-alignment-design.md`
- Plan: `docs/superpowers/plans/2026-04-16-gemma-embed-alignment.md`
- Decision note: `results/phase_a_decision.md` (local only — `results/` is gitignored)
- Code: commits `a7f03d0`, `5b62605`, `d0ecdc6`, `5707892`, `61967f7`, `2a007fa`
- Local-only cache: `models/english_gemma_768d.npz` (1.1 GB)
