# Cuneiformy Roadmap

Queued workstreams, ordered by priority. Each item is shippable on its own; items lower on the list assume items above have landed.

For historical context on what's already shipped, see [`EXPERIMENT_JOURNAL.md`](EXPERIMENT_JOURNAL.md).

---

## 1. Hyper-glyphy monorepo reorganization

**Status:** queued, not started.
**Estimated scope:** ~1 week (Phase α only — structural move).
**Prerequisite:** none.
**Unblocks:** everything below item 2.

Rename the repo from `cuneiformy` → `hyper-glyphy` and reorganize to host multiple ancient-language embedding spaces as siblings under one framework. Target structure:

```
hyper-glyphy/
├── languages/
│   ├── sumerian/         # current Cuneiformy content
│   ├── egyptian/         # Phase β: port Heiroglyphy + Gemma target
│   ├── akkadian/         # future
│   ├── greek/            # future
│   ├── hattusian/        # future
│   └── jiaguwen/         # future (oracle bone / early Chinese)
├── framework/            # civilization-agnostic: anomaly_lenses, anomaly_framework,
│                         # shared lookup base, alignment pipeline skeleton
├── shared/               # shared English-target artifacts (whitened Gemma, GloVe)
├── comparative/          # cross-civilizational analyses spanning >1 language
└── docs/                 # repo-wide research vision, journal, roadmap
```

**Work required (Phase α only):**
- `git mv` all Sumerian-specific files into `languages/sumerian/`.
- Extract civilization-agnostic framework code to `framework/`.
- Move shared English-target artifacts (whitened Gemma cache, GloVe) to `shared/`.
- Update all import paths and hardcoded references.
- Re-run full pipeline + regenerate artifacts in new locations.
- Rename GitHub repo; update README.
- Verify 167 tests still pass post-move.

**What this does NOT include:** the Egyptian sibling-language build. That's Phase β (~2 weeks, depends on this one).

**Why do it before the second language:** retrofitting structure onto two civilizations is harder than moving one and adding a second. Reorganizing before the second language is committed is cleaner than after.

---

## 2. Egyptian sibling language (Phase β)

**Status:** queued, blocked on item 1.
**Estimated scope:** ~2 weeks.
**Prerequisite:** Hyper-glyphy reorg landed (Phase α).

Port the Heiroglyphy V15 alignment pipeline into `languages/egyptian/`, targeting whitened Gemma (new for Egyptian — Heiroglyphy currently only aligns to GloVe). Produce:
- `languages/egyptian/final_output/egyptian_aligned_gemma_vectors.npz`
- `EgyptianLookup` class mirroring `SumerianLookup`'s dual-view API
- Egyptian atlas via the framework's reusable `anomaly_atlas` infrastructure
- Egyptian cosmogony case study mirroring the Sumerian one (suggested concepts: `mꜣꜥt`, `nṯr`, `kꜣ`, `ḫt-nṯr`, `ꜣḫ`)

---

## 3. Comparative cross-civilizational document (Phase γ)

**Status:** queued, blocked on item 2.
**Estimated scope:** ~1 week after both civilizations have atlases.
**Prerequisite:** Egyptian language module shipped.

Single document comparing Sumerian and Egyptian atlases side-by-side. Same cuneiform-font-template infrastructure renders Egyptian hieroglyphic signs alongside cuneiform.

Anchor questions:
- Which concepts have universals (low displacement in both languages)?
- Which have culture-specific geometric structure (high displacement in one language, low in the other)?
- Are the theme patterns (translation failures, grammatical bridges, specialized cultic vocabulary) cross-civilization stable?

---

## 4. Additional ancient languages

**Status:** queued, prioritized by corpus availability.

In approximate order of feasibility:

- **Akkadian** — well-documented, overlapping with Sumerian corpus period, ORACC coverage good. Similar difficulty to Sumerian; ~1-2 weeks.
- **Classical Greek** — Perseus Digital Library provides a clean corpus with extensive translations; GloVe/Gemma have strong English coverage of Greek-descended vocabulary. ~1 week.
- **Hattusian (Hittite)** — smaller but clean corpus via Hittite Dictionary project. ~1-2 weeks.
- **Jiaguwen (oracle bone)** — hardest. Fragmentary corpus, interpretive scholarship varies, no strong Classical-Chinese English-alignment baseline to ground against. ~3-4 weeks; corpus assembly dominates the cost.

---

## 5. Anchor semantic-quality workstream (Sumerian-specific)

**Status:** queued, can run in parallel with 1–4.
**Estimated scope:** ~3–5 days.
**Prerequisite:** none; runs against current Sumerian pipeline.

The anomaly atlas's Lens 1 unfiltered and the §3 "Translation failures" section of the interpretive document both surfaced high-confidence anchor pairs with negative cosine similarity (e.g., `rin2 → lord`, `sirara → c`). Triage:

- Remove anchors with degenerate English glosses (single letters, numerics-only, named-entity clashes).
- Consider downgrading confidence on anchors with cosine similarity < 0 to their English glosses (they're either ePSD2 glosses that don't capture how the word actually distributes, or genuine translation errors in the dictionary).
- Re-run alignment; measure whether top-1 accuracy moves (expected: small improvement, +1-2pp — this is polish, not a breakthrough).

---

## 6. Steering / interpretability bridge (optional research track)

**Status:** deferred. Speculative research direction.
**Estimated scope:** ~3-4 weeks.
**Prerequisite:** none specific; can happen any time.

Train a linear projection from our 768d whitened-Gemma Sumerian space → Gemma 3 / Gemma 4 LLM residual stream. Use contrastive Sumerian concept pairs (e.g., primordial vs. decreed) as CAA-style steering vectors. Test whether ancient-language semantic axes have mechanistic traction inside modern LLMs.

Result could be positive (ancient-language alignment is a novel source of interp vectors) or negative (modern LLMs don't linearly encode Sumerian-specific distinctions). Either is publishable.

---

## Shipped (for reference)

See [`EXPERIMENT_JOURNAL.md`](EXPERIMENT_JOURNAL.md) for dated entries. In chronological order, this repo has shipped:

1. v1 baseline — 17.30% top-1 GloVe (direct port of Heiroglyphy V15).
2. Phase A whitening — BERT-whitening for EmbeddingGemma target; +2.54pp top-1.
3. Phase B dual-view — both GloVe and whitened Gemma as parallel targets.
4. Workstream 2a — anchor audit baselining valid-anchor rate at 14.05%.
5. Workstream 2b-pre — coverage diagnostic attributing 64.85% of dropout to ASCII normalization.
6. Workstream 2b — normalization fix; top-1 19.85% → 52.13% (stretch tier).
7. Sumerian cosmogony document — 14,026-word case study on five Anunnaki-related concepts.
8. Anomaly atlas — civilization-agnostic diagnostic framework + six-lens atlas over 35,508 tokens.
9. Anomaly atlas interpretive document — 9,896-word thematic interpretation + PDF with embedded cuneiform font.
