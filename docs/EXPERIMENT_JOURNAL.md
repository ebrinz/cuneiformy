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
