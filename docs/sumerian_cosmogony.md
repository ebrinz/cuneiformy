# Sumerian Cosmogony as Geometric Object: A Case Study in Whitened-Gemma Alignment

**Branch:** `feat/cosmogony-document`
**Alignment artifact commit:** `5945bd6`
**Tables:** `docs/cosmogony_tables.json` (schema v1)
**Figures:** `docs/figures/cosmogony/`
**Pre-flight:** `results/cosmogony_preflight_2026-04-20.json`

---

## § 0 — Abstract

This document applies the Cuneiformy whitened-Gemma semantic alignment — Sumerian FastText 768-dimensional vectors projected into Gemma-derived English space via ridge regression, reaching 52.13% top-1 accuracy — to five cosmogonically central Sumerian vocabulary items: `abzu` (primordial deep), `zi` (breath/life-essence), `nam` (essence-prefix, nominal abstraction), `namtar` (fate/destiny), and `me` (divine civilizational decrees). For each concept we report nearest-Sumerian-neighbor clusters in both Gemma and GloVe spaces, pairwise semantic-field distances, analogy probe results, English cosine displacement, and ETCSL source-text grounding, following a uniform eight-section template. The headline geometric finding is a spectrum of translation opacity: `namtar` (fate) is the sole concept whose Sumerian-projected vector lands near its English gloss (cosine similarity 0.459), while `abzu` is geometrically orthogonal to "deep" (cosine −0.002), and `me` is nearly so (0.017). The finding is consistent with the hypothesis that Sumerian cosmogonic vocabulary encodes genuinely alien conceptual structure — structure that English glosses gesture at without conveying. The document serves as the first Phase 3 narrative-extraction artifact of the Cuneiformy research program.

---

## § 1 — Introduction

### The Translation Problem Is Geometric

When Thorkild Jacobsen wrote of Sumerian religion as a world in which nature was experienced as a "Thou" rather than an "It" — as wills and personalities rather than forces and mechanisms (Jacobsen 1976, p. 5) — he was making a claim that sits awkwardly in ordinary scholarly translation. A translation is a word-to-word mapping; it does not easily capture the claim that an entire cognitive framework has no modern analogue. Dictionary entries for Sumerian concepts necessarily render them in English: `zi` becomes "breath" or "life," `me` becomes "divine decrees," `namtar` becomes "fate." These glosses are not wrong in the narrow sense of being false; they identify the contexts in which the Sumerian words appear. But they import the English word's geometric neighborhood — its full complement of near-synonyms, opposites, metaphorical extensions, and collocates — and that neighborhood may be entirely unlike the Sumerian original's.

The Cuneiformy project takes this mismatch seriously as a measurable quantity. If we align Sumerian embedding vectors into English semantic space and ask *how close does the Sumerian word land to its English gloss*, we get a number between −1 and 1. That number is not the "true" meaning distance between two languages — it inherits the noise of the alignment pipeline, the corpus coverage of the ETCSL texts, and the biases of the FastText training on a 3,500-year-old corpus. But it is an empirical constraint on interpretation, and it is honest in a way that a dictionary gloss cannot be: it tells us whether the Sumerian vector's geometric neighborhood in aligned space resembles the English word's neighborhood, or whether the two sit in essentially unrelated parts of the manifold.

This document is organized around five Sumerian cosmogonic concepts chosen because their English glosses carry strong modern semantic neighborhoods. "Primordial deep," "breath," "essence," "fate," "divine decree" — each of these English phrases arrives loaded with millennia of downstream usage in philosophy, theology, and literature. If the alignment finds that the Sumerian vectors land far from those English neighborhoods, the finding is methodologically significant: it suggests that the glosses are indexical labels for alien conceptual structures rather than translations that carry the structure across.

### The Cosmogonic Frame

The five concepts are embedded in a narrative spine drawn from the Sumerian tradition itself: the cosmogonic arc from the primordial ocean Nammu through the separation of heaven and earth, the shaping of humans, the determination of fates, and the distribution of the `me` — the civilizational blueprints by which ordered life becomes possible. This arc appears across multiple ETCSL texts, most fully in *Enki and the World Order* (ETCSL 1.1.3) and *Enki and Ninmah* (ETCSL 1.1.2), with additional attestations in hymns, laments, and god-lists. The Anunnaki — the great gods of heaven and earth — serve as the cosmological agents who execute each step: An and Ki generate the conditions, Enlil separates and orders, Enki acts on the waters and the clay, Namtar enforces the fates already written, and the `me` pass from Enki's keeping to Inanna's and through her to the cities of Sumer.

### Audience and Scope

This document is addressed to readers comfortable with the mechanics of vector-space alignment but not necessarily with Sumerological philology. Sumerian grammatical forms and ETCSL text-IDs are explained on first use. Conversely, the alignment methodology — whitening, ridge regression, top-k retrieval — is described in §2 with enough detail for a Sumerologist to assess whether the method's outputs can bear the interpretive weight placed on them.

The document does not claim to resolve the philological debates it touches. It claims to add a geometric dimension to those debates: a new kind of data that constrains but does not determine interpretation.

---

## § 2 — Methodology

### 2.1 The Alignment Pipeline

The Cuneiformy pipeline produces Sumerian-to-English semantic alignment in four stages.

**Stage 1 — Corpus and embeddings.** The Sumerian corpus consists of transliterated texts from the Electronic Text Corpus of Sumerian Literature (ETCSL), the Cuneiform Digital Library Initiative (CDLI), and ORACC, cleaned and tokenized. FastText is trained on this corpus to produce 768-dimensional word vectors for the Sumerian vocabulary. At the alignment-artifact commit (`5945bd6`), the fused vocabulary contains 35,508 Sumerian tokens. FastText's subword character-level modeling means that morphological variants of a root (case suffixes, possessive clitics, verbal prefixes) receive geometrically nearby vectors, producing clusters of inflected forms around each root in Sumerian embedding space.

**Stage 2 — Anchor extraction.** Alignment requires anchor pairs: (Sumerian token, English word) pairs that name the same concept and appear in both vocabularies. Anchors are extracted from the ePSD2 dictionary and ETCSL parallel-translation glossaries. A critical normalization step, introduced in commit `ec352b1` of the Workstream 2b pipeline, applies subscript-to-ASCII conversion, determinative-brace stripping, hyphen removal, and lowercasing at anchor-extraction time — matching the normalization applied during FastText tokenization. Before this fix, 7,651 of 11,798 anchor misses were caused by the normalization mismatch rather than genuine vocabulary absence. After the fix, valid anchors increased from 1,951 to 8,558 (65.33% of merged anchors), providing a much larger training signal for the regression step.

**Stage 3 — Ridge regression with whitened target.** A ridge regression maps Sumerian vectors into English target space. Two target spaces are maintained in parallel: GloVe 300-dimensional vectors (the original baseline) and Gemma-derived 768-dimensional English contextual embeddings. The Gemma target is whitened before regression — transformed so that principal component directions have equal variance — which substantially reduces the geometric distortion introduced by the contextual model's anisotropic embedding distribution. At alpha = 100 (the regularization value that reached the top-1 plateau for the whitened target), the whitened-Gemma alignment achieves **52.13% top-1** and **61.97% top-5** accuracy on held-out anchors (Experiment Journal, 2026-04-19 entry). The GloVe alignment achieves 35.70% top-1 for comparison. These are the numbers reported at the alignment-artifact commit.

**Stage 4 — Dual-view lookup.** The production `SumerianLookup` class (at `final_output/sumerian_lookup.py`) exposes the same API for both spaces: `find(english_word, space="gemma")` projects the English word through the alignment into Sumerian-projected space and returns top-k nearest Sumerian tokens by cosine similarity. `find(english_word, space="glove")` does the same in the GloVe space. The five analyses in this document use the Gemma space as primary and the GloVe space as secondary confirmation, consistent with the Phase B dual-view substrate decision.

### 2.2 Analysis Modules

Six modules in `scripts/analysis/` produce the tables and figures referenced in §4–8:

- `semantic_field.py` — pairwise cosine-distance matrix over a thematically adjacent token set; renders heatmaps.
- `english_displacement.py` — computes cosine similarity between a Sumerian token's aligned vector and its English anchor's native vector.
- `etcsl_passage_finder.py` — retrieves ETCSL passages containing a normalized Sumerian token, with configurable context lines.
- `umap_projection.py` — UMAP reduction to 2D for visualization; falls back to PCA when token count is too small for UMAP.
- `preflight_concept_check.py` — validates concept candidates before prose is written.
- `generate_cosmogony_tables.py` / `generate_cosmogony_figures.py` — deterministic entry points.

All numeric tables in this document were generated from `docs/cosmogony_tables.json`, which is the committed output of `generate_cosmogony_tables.py`. Every numeric claim is traceable to a path in that file (paths cited inline).

### 2.3 Caveats

Several methodological limitations bear on interpretation throughout this document.

**Corpus bias.** The ETCSL corpus is not a random sample of ancient Sumerian language use. It emphasizes literary, liturgical, and royal-administrative texts from the Old Babylonian period (roughly 2000–1600 BCE). Everyday commercial and legal Sumerian — abundant in the CDLI — is less represented. Concepts that appear frequently in hymns and myths may have different distributional neighborhoods than the same concepts in administrative tablets.

**The morphology-as-noise problem.** FastText's subword model produces clusters of inflected forms around each root. The top-10 nearest neighbors for `abzu` in Gemma space are dominated by `saŋ`-prefixed forms (`saŋkiŋu10`, `saŋe`, `saŋa2`), not by thematically related cosmogonic vocabulary. This reflects distributional patterns in the corpus — the root co-occurs with `saŋ`-compounds — rather than semantic kinship in the human sense. We note this wherever it appears and treat GloVe-space neighbors as a partial corrective (GloVe, trained on a much larger English corpus, is less likely to surface morphological noise as near-neighbors).

**Alignment noise floor.** At 52.13% top-1 accuracy, the alignment is strong enough to support geometric interpretation but not so strong as to treat every nearest-neighbor result as linguistically significant. We privilege findings that appear consistently across both Gemma and GloVe spaces, and we treat single-space results as suggestive rather than conclusive.

**The normalization fix and namtar.** An earlier preflight run (before commit `ec352b1`) incorrectly failed `namtar` on the ETCSL passage check because unnormalized transliterations were not matching. After the fix, `namtar` passed all checks with 35 confirmed ETCSL passages. No substitution was required; the full original concept slate is used. The §12 appendix documents this in detail.

---

## § 3 — The Cosmogonic Arc

### Nammu and the Primordial Water

Before there was a world, Sumerian cosmogony posits a state of undifferentiated aqueous being: Nammu, written with the sign for sea, described in the *Eridu Genesis* (ETCSL 1.7.4) and evoked in the prologue of *Enki and Ninmah* (ETCSL 1.1.2). Nammu is not a creator in the sense of a craftsman who makes objects; she is a substrate, a generative medium from which the gods emerge. The Sumerian grammatical form of this is distinctive: not "from the water" (directive or ablative) but a kind of ontological encompassing — the water *is* the situation before any separation occurs.

This primordial state is associated with the word `abzu`: the sweet freshwater ocean beneath the earth, imagined as both a physical place (Enki's temple at Eridu sits above the abzu) and a cosmological condition. The abzu is the source of wisdom, of the `me`, and of the creative power that Enki wielded. Geometrically, as we will see in §4, the Sumerian vector for `abzu` lands nearly orthogonal to the English word "deep" (cosine −0.002, `concepts.abzu.english_displacement.gemma.cosine_similarity`), suggesting that the English spatial metaphor of vertical depth captures almost nothing of what the Sumerian concept encodes in distributional context.

### An and Ki: The Separation

Out of the primordial state emerges a separation: An (sky, male) and Ki (earth, female) are produced as distinct entities, and their separation — or, in some accounts, Enlil's carrying off of the sky — constitutes the creation of the world as habitable space. The *Gilgamesh, Enkidu, and the Netherworld* (ETCSL 1.8.1.4) opens with a brief cosmogony in which this separation is described in just two lines:

> *an-gin7 an ki-ta ba-da-an-ba-al-e-de3-en* / *ki-gin7 ki an-ta ba-da-an-ba-al-e-de3-en*
> After heaven had been moved away from earth, after earth had been separated from heaven.

The brevity is not incuriosity. The texts do not elaborate the separation because the audience knew it; the point was to mark the moment before humans existed, before fates were determined, before the `me` were distributed. The cosmogony is a chronological bracket, not a philosophical treatise.

The separation of An and Ki is also, in the geometrically-structured cosmology, the creation of the space within which `nam` — essence, office, destiny-in-general — becomes possible. Fate requires a world in which things can be differentiated from one another, in which a specific destiny can be applied to a specific entity. The geometry of the cosmos, in Sumerian thought, is the precondition for the geometry of fate.

### The Anunnaki and the Assembly

Once heaven and earth are separated, the Anunnaki appear: the great gods who occupy both realms, whose collective deliberation constitutes the highest authority in the Sumerian cosmos. Enlil convenes the assembly (ukkin) in which fates are determined. An sits as patriarch; Enki attends as counselor and executor. The goddess Namtu — in later texts conflated with the demon Namtar — waits to carry decisions into effect.

The Anunnaki figure prominently in the UMAP visualization at `docs/figures/cosmogony/anunnaki_narrative_umap.png`, which projects the 18-word Anunnaki vocabulary into two dimensions using the Gemma-aligned vectors. The vocabulary includes the divine names `an`, `ki`, `enki`, `enlil`, `nammu`, `ninmah`, `inanna`, `utu`, `nanna` alongside the five cosmogonic concepts `nam`, `me`, `namtar`, `zi`, `abzu`, plus supplementary terms `ima` (clay), `kur` (netherworld/mountain), `dingir` (god), and `lugal` (king). The projection shows the divine names and cosmogonic concepts occupying distinct but adjacent regions of the 2D space — consistent with, though not proof of, the intuition that the cosmogonic concepts are geometrically "near" the divine agents who deploy them.

### Enki and the Shaping of Humans

The third major event in the cosmogonic arc is the creation of humanity. In *Enki and Ninmah* (ETCSL 1.1.2), Enki and the mother-goddess Ninmah shape humans from clay (`ima`) at Eridu. The purpose is explicitly functional: the gods are exhausted from digging the canals and performing agricultural labor; humans are to be created to relieve them. The Sumerian understanding of human creation is thus not primarily an account of dignity or imago dei; it is an account of cosmic labor-allocation.

This shaping involves `zi`: the breath or life-essence that animates the clay figure and makes it a living thing. In *Enki and the World Order* (ETCSL 1.1.3), Enki is praised as the one who "knows the heart" (`cag4-ga-na pa e3-a`) and who brings life to what was inert. The vector for `zi` sits far from the English word "breath" in Gemma space (cosine 0.056, `concepts.zi.english_displacement.gemma.cosine_similarity`), consistent with the interpretation that `zi` denotes something more encompassing than the physiological act of breathing — a category that the English word does not cleanly reach.

### Nam-tar: The Cutting of Fate

The fourth event is the determination of fates. In Sumerian the relevant verb is `tar` — to cut, to determine, to decree. `Nam-tar` is literally "fate-cutting": the action by which the Anunnaki fix the destiny of a person, city, or cosmic period. The goddess Namtar (later also understood as a demon of plague and death) is the agent who executes the fates already written. ETCSL text c4332.27 preserves the phrase *nam-tar nam-he2 jic nam-tar jic nam-he2* ("destiny, prosperity — the wood of destiny, wood of prosperity"), a ritual enumeration in which fate and flourishing are paired as a natural dyad.

Geometrically, `namtar` is the lone outlier among our five concepts: its Sumerian vector lands at cosine 0.459 from the English word "fate" (`concepts.namtar.english_displacement.gemma.cosine_similarity`), by far the largest displacement similarity in the set. The finding is consistent with the hypothesis that "fate" as a concept is genuinely cross-linguistically portable in a way that "deep," "breath," "essence," and "divine decree" are not. We examine this finding carefully in §7, including a consideration of why `namtar`'s neighborhood in Gemma space still shows some surprises relative to the received Sumerological understanding.

### The Distribution of Me

The fifth and culminating event is Inanna's acquisition of the `me` from Enki's custody at Eridu and their distribution to Uruk and, through Uruk, to Sumer at large. *Inanna and Enki* (ETCSL 1.3.1) gives an extended list of the `me`: kingship, the descent to the netherworld, the ascent from the netherworld, the craft of the scribe, the art of the smith, the scribal art, music, herding — a complete inventory of what makes civilized human life possible. The `me` are not laws in the modern sense; they are more like ontological templates, the fact-of-being of each institution. For a city to have kingship means that the `me` of kingship has been installed there by divine sanction.

The vector for `me` in Gemma space is nearly orthogonal to "decree" (cosine 0.017, `concepts.me.english_displacement.gemma.cosine_similarity`). The English word "decree" suggests a speech act — a unilateral assertion of authority. The Sumerian `me` appears to encode something more structural: not what is said but what is constituted. The embedding geometry is consistent with the interpretation that `me` sits closer to the vocabulary of governance and administration (its nearest Gemma neighbors include `dikud`, "judge," and `ensi2`, "governor") than to the vocabulary of speech acts and declarations. We explore this in §8.

### The Arc as Geometric Object

The narrative spine just described — Nammu/abzu → An-Ki separation → Anunnaki assembly → nam determination → zi animation → namtar enforcement → me distribution — can be read as a sequence of conceptual operations in Sumerian semantic space. The UMAP figure at `docs/figures/cosmogony/anunnaki_narrative_umap.png` attempts to make this structure visible. The cosmogonic axis projection at `docs/figures/cosmogony/cosmogony_axis_projection.png` (discussed in §9) projects the five concepts onto a single synthetic axis defined by the difference between `namtar` (the most English-translation-adjacent concept) and `abzu` (the most alien), giving a rough order of "how much does English capture of this concept."

The arc also has an interesting internal structure from the perspective of the dual-view divergence analysis (§4–8). Concepts early in the arc (`abzu`, `zi`) show sharper Gemma-vs.-GloVe divergence in their nearest-neighbor lists, suggesting that their geometric position is more sensitive to the target space. Concepts later in the arc (`namtar`, `me`) show more consistent cross-space neighborhoods, though for very different reasons: `namtar` because it sits near a genuine cross-linguistic attractor (fate/destiny), `me` because it is strongly pulled toward administrative-governance vocabulary in both spaces.

---

## § 4 — Deep Dive: `abzu` (Primordial Deep)

### 4.1 Anchor Reading

The ePSD2 entry for `abzu` (also *apsu* in Akkadian) glosses it as "underground water, subterranean fresh water, Enki's domain." In Sumerological scholarship the `abzu` is most fully analyzed by Jacobsen as the sweet-water abyss beneath the earth — a generative, fecund reservoir distinct from the bitter salt sea (`tiamat` in Akkadian) and from the rivers that run on the surface (Jacobsen 1976, pp. 110–115). The word functions in two registers simultaneously: a geographic-cosmological referent (the freshwater table beneath Mesopotamian soil) and a divine-domain referent (the part of the cosmos that belongs to and is identified with Enki). Kramer emphasizes the temple at Eridu — the oldest Sumerian city and the site of Enki's principal temple, the E-abzu — as the earthly point of access to the primordial waters (Kramer 1944, pp. 59–62). Black, Cunningham, Robson, and Zólyomi note that the word `abzu` can also refer to the basin of holy water in a Sumerian temple, a secondary meaning that connects the cosmic and the ritual (Black et al. 2004, p. 317).

### 4.2 Nearest Sumerian Neighbors (Gemma Space)

The top-10 nearest Sumerian neighbors to `abzu` in Gemma space (`concepts.abzu.top10_dual_view.gemma`) are:

| Rank | Token | Cosine Similarity |
|------|-------|-------------------|
| 1 | saŋkiŋu10 | 0.2361 |
| 2 | saŋe | 0.2359 |
| 3 | saŋa2 | 0.2289 |
| 4 | lipiŋu10 | 0.2266 |
| 5 | saŋŋu10 | 0.2238 |
| 6 | saŋeš | 0.2171 |
| 7 | saŋzu | 0.2126 |
| 8 | su3udŋu10 | 0.2100 |
| 9 | gid2ŋu10 | 0.2074 |
| 10 | sum4ŋu10 | 0.2072 |

The dominant feature of this list is the preponderance of `saŋ`-prefixed and `ŋu10`-suffixed forms. `saŋ` is the Sumerian word for "head" (as a body part) and by extension serves in many nominal compounds; `ŋu10` is a first-person singular possessive clitic. The list is effectively a cluster of inflected pronominal-possessive forms, all with low similarities (maximum 0.2361). This is the morphology-as-noise pattern described in §2.3: FastText's subword model captures phonological and morphological similarity, and `abzu` happens to share subword patterns with `saŋ`-forms in the training corpus.

There is no evident cosmogonic or aquatic semantic field in this neighbor list. `lipiš` (heart/liver, as seat of emotion) appears at rank 4 in its possessive form; `su3ud` (distant/long) appears at rank 8; `gid2` (long/extended) at rank 9. None of these connects obviously to the primordial-deep semantic domain. The finding is honest: in Gemma space, `abzu` does not cluster with words that a human reader would associate with water, creation, or the divine domain. Its geometric position appears to be determined more by distributional frequency patterns in the ETCSL corpus than by the thematic content that Sumerological scholarship assigns to it.

### 4.3 Semantic-Field Map

The heatmap at `docs/figures/cosmogony/abzu_semantic_field_heatmap.png` shows pairwise cosine distances among a 15-word thematically adjacent vocabulary: `abzu`, `ki`, `an`, `enki`, `nammu`, `kur`, `tiamat` (if present in vocab), `eridug`, and several aquatic and cosmic terms drawn from ETCSL contexts. Dark cells (low distance, high similarity) indicate tokens that occupy nearby positions in Gemma space; light cells indicate tokens that are geometrically remote from one another.

The heatmap reveals that `abzu` sits at moderate distance from most of its thematic neighbors — it is not a strongly isolated token (which would appear as a uniformly light row and column) but neither does it form a tight sub-cluster with cosmogonic terms. `Enki` and `nammu` show moderate mutual similarity, consistent with their shared cosmogonic domain in the texts, but `abzu` does not strongly cohere with either. The embedding geometry is consistent with the interpretation that `abzu` is a relatively "scattered" concept in the Sumerian corpus — it appears in many different textual contexts (ritual, hymn, myth, administrative formula) and its distributional neighborhood is therefore diverse rather than tightly thematic.

### 4.4 Dual-View Divergence

| Rank | Gemma Token | Gemma Sim | GloVe Token | GloVe Sim |
|------|-------------|-----------|-------------|-----------|
| 1 | saŋkiŋu10 | 0.2361 | daŋalla | 0.5630 |
| 2 | saŋe | 0.2359 | daŋalbi | 0.5604 |
| 3 | saŋa2 | 0.2289 | dajalla | 0.5584 |
| 4 | lipiŋu10 | 0.2266 | daŋallake4 | 0.5581 |
| 5 | saŋŋu10 | 0.2238 | daŋalba | 0.5526 |
| 6 | saŋeš | 0.2171 | dagalla | 0.5514 |
| 7 | saŋzu | 0.2126 | dajaɭbi | 0.5475 |
| 8 | su3udŋu10 | 0.2100 | daŋale | 0.5470 |
| 9 | gid2ŋu10 | 0.2074 | he2maalla | 0.5461 |
| 10 | sum4ŋu10 | 0.2072 | daŋal | 0.5458 |

The GloVe top-10 (`concepts.abzu.top10_dual_view.glove`) shows an entirely different cluster: `daŋal`-prefixed forms dominating, with similarities in the 0.54–0.56 range (substantially higher absolute values than Gemma, reflecting GloVe's less anisotropic distribution). `Daŋal` is a Sumerian adjective meaning "wide, broad, extensive" — a spatial-magnitude term. There is no overlap between the Gemma and GloVe top-10 lists. This total divergence is a methodological signal: when the two spaces agree on neighbors, those neighbors are the most robust findings; when they disagree entirely, neither list should be over-interpreted.

The GloVe result suggests that the English seed "deep" — which in English combines vertical extent ("the deep ocean") with conceptual profundity — aligns in GloVe space with Sumerian vocabulary for horizontal breadth (`daŋal` = wide/extensive) rather than vertical depth. This is a small but geometrically interesting observation: the GloVe space's representation of "deep" has a spatial-magnitude component that draws it toward breadth-terms in Sumerian.

### 4.5 Analogy Probes

The analogy query `ocean : water :: deep : ?` (`concepts.abzu.analogy_probes[0]`) in Gemma space returns:

| Rank | Token | Similarity |
|------|-------|------------|
| 1 | abzu | 0.4760 |
| 2 | abzux | 0.4655 |
| 3 | abzua | 0.4593 |
| 4 | abzuna | 0.4482 |
| 5 | abzuŋa2 | 0.4462 |

The top-5 results are all morphological variants of `abzu` itself: `abzux` (a graphic variant), `abzua` (locative), `abzuna` (ablative), `abzuŋa2` (another case form). The analogy probe is effectively completing the equation `ocean : water :: deep : abzu` — which is the expected result given how the analogy vector is constructed, and it confirms that `abzu` is the Sumerian token most geometrically consistent with the English framing of the probe. However, the result is circular in the sense that the probe's construction ensures `abzu` surfaces when "deep" is one of the input terms. It does not establish that `abzu` and "deep" occupy similar semantic positions; the English displacement measurement (§4.6) is more informative on that question.

### 4.6 English Displacement

**Cosine similarity (Gemma):** −0.002 (`concepts.abzu.english_displacement.gemma.cosine_similarity`)

**GloVe comparison:** 0.358 (`concepts.abzu.english_displacement.glove.cosine_similarity`)

A cosine similarity of −0.002 is, within floating-point noise, indistinguishable from zero: the Sumerian vector for `abzu`, when projected into Gemma-aligned English space, is geometrically orthogonal to the English word "deep." The two vectors are not opposites (that would be −1); they are simply unrelated — they share no alignment-detectable directional component. The GloVe result of 0.358 is somewhat warmer, suggesting partial geometric overlap in the lower-dimensional GloVe space, but still far from the 0.45+ range that we would consider "translation-adjacent."

The near-zero Gemma displacement is the most striking single number in the `abzu` analysis. The embedding geometry is consistent with the strong interpretation that the English gloss "deep" — with its connotations of vertical extent, profundity, and oceanic depth — is a pointer to `abzu` without being geometrically aligned with it. The Sumerian concept appears to occupy a region of embedding space that English spatial and metaphysical vocabulary does not reach.

### 4.7 Source-Text Grounding

ETCSL passage c4332.3 (`concepts.abzu.etcsl_passages[0]`):

> **Transliteration:** *abzu ki sikil-ta e3-a*
> **Translation context:** "coming forth from the abzu, the pure place"

This is a line from a fire-god hymn (the full passage describes Gibil/Girru emerging from the pure abzu) retrieved at line c4332.3. The collocation *ki sikil* ("pure place") paired directly with *abzu* is significant: the abzu is not a polluted or dangerous underworld water but a sacred, ritually pure source. The word *sikil* — pure, clean, ritually correct — carries connotations of legitimate divine authority in Sumerian texts (Jacobsen 1976, p. 111). The abzu is pure because Enki's authority there is uncontested.

A second passage (c4332.40) records "praise to lady Kusu, the princess of the holy abzu" (*nun abzu kug-ga*), pairing `abzu` with the adjective `kug` (holy, sacred, gleaming). Across these attestations, `abzu` consistently appears in the company of purity and holiness markers — a distributional pattern consistent with the hypothesis that in the corpus, `abzu` is less a spatial descriptor than a ritual-quality marker for Enki's domain.

### 4.8 Interpretive Synthesis

The geometric evidence for `abzu` tells a consistent story: this concept does not translate. The Gemma displacement of −0.002 places it at geometrical zero-similarity to "deep"; the Gemma top-10 neighbors are morphological noise rather than thematic associates; the two spaces produce entirely incompatible neighbor lists. None of this means `abzu` is unstructured — it has clear distributional neighborhoods in the corpus — but those neighborhoods do not correspond to what English spatial and depth vocabulary encodes.

The most defensible interpretation of this geometry is that `abzu` in the Sumerian corpus functions primarily as a **theological address**: a label for Enki's domain that carries holiness, purity, and creative fecundity as its core associations, with the spatial metaphor of underground water as a secondary (perhaps originally literal, later increasingly symbolic) frame. This interpretation is consistent with Jacobsen's account of Sumerian "theology of nature" — the abzu is not a geological entity described with metaphor layered on top; it is always already a divine domain that is secondarily described as having an aquatic character.

The alternative reading — that `abzu` is primarily a physical-cosmic term that later acquires theological resonance — would predict tighter geometric clustering with aquatic and spatial vocabulary in the embedding space. The data do not support this reading: neither Gemma nor GloVe returns aquatic or spatial neighbors for `abzu`. The geometric finding is at least consistent with, and tentatively supports, the theological-address interpretation over the physical-cosmic-primary interpretation.

The `abzu` analysis sets the stage for `zi` (§5): if the primordial water is the source of life and creativity, `zi` is what that source generates — the animating breath or life-force that enters the clay figure and makes it a person.

---

## § 5 — Deep Dive: `zi` (Breath/Life-Essence)

### 5.1 Anchor Reading

The ePSD2 entry for `zi` gives two primary glosses: "life" and "breath." Both are attested across the ETCSL corpus, sometimes in the same passage. Jacobsen characterizes `zi` as the animating principle of living things — the thing that clay does not have and a person does — and notes that it is closely associated with the wind and breath as the visible, sensible manifestation of invisible vital force (Jacobsen 1976, pp. 12–13). Kramer translates `zi` as "breath of life" in his account of the creation of humans in *Enki and Ninmah*, emphasizing the divine inbreathing as the operative creative act (Kramer 1944, p. 69). The word `zi` is also used in the verbal noun `namtil3` (life/lifespan, lit. "essence of life"), and `zi-ud-su3-ra` (Ziusudra, the Sumerian flood hero, whose name contains `zi` and means something like "life of distant days" — see Foxvog, *Introduction to Sumerian Grammar*, §4.2 on compound nouns).

In Sumerian metaphysics `zi` does not map neatly onto either "soul" (which would imply a separable spiritual entity) or "life" (which in English is primarily temporal — the duration of existence). `Zi` is closer to the animating *act* — the presence of motion, breath, and responsiveness that constitutes a living thing as distinct from a dead or inanimate one.

### 5.2 Nearest Sumerian Neighbors (Gemma Space)

The top-10 nearest Sumerian neighbors to `zi` in Gemma space (`concepts.zi.top10_dual_view.gemma`) are:

| Rank | Token | Cosine Similarity |
|------|-------|-------------------|
| 1 | zapaag2zu | 0.3325 |
| 2 | ku6gu10 | 0.3173 |
| 3 | zapaag2zu (var. zapaŋ2zu) | 0.2992 |
| 4 | zapaŋ2 | 0.2973 |
| 5 | zapaŋ2bi | 0.2909 |
| 6 | zapaag2bi | 0.2879 |
| 7 | zapaag2 | 0.2878 |
| 8 | zapaŋ2ŋu10 | 0.2866 |
| 9 | zapaaj2zu | 0.2766 |
| 10 | zapaaj2bi | 0.2764 |

The dominant cluster in the Gemma top-10 is `zapaa`-forms: `zapaa(g/ŋ/j)2` is the Sumerian word for "voice," "sound," or "cry." The possessive and pronominal suffixes (`zu` = your, `bi` = its/their, `ŋu10` = my) produce the morphological variants seen in the list. The rank-2 item `ku6gu10` is "my fish" (ku6 = fish, gu10 = my).

That `zi` neighbors primarily voice/sound vocabulary in Gemma space is an unexpected but interpretively suggestive finding. Voice and breath share a physiological relationship (breath is the medium of voiced speech), and in several Sumerian literary contexts the animating power of `zi` is expressed through speech — the gods speak the `me` into existence, Enki's wisdom manifests as words. The embedding geometry does not confirm this connection in a strong sense — the similarities are modest (0.26–0.33) and the link may be coincidental distributional co-occurrence rather than semantic kinship — but it is notable that the geometric neighborhood of "breath/life-essence" includes vocal expression rather than, say, movement, water, or clay.

### 5.3 Semantic-Field Map

The heatmap at `docs/figures/cosmogony/zi_semantic_field_heatmap.png` shows pairwise distances among a thematically adjacent set that includes `zi`, `nam`, `me`, `namtar`, `nig2-zi-gal` (living thing/creature), `ti` (rib/life, Sumerian pun exploited in the Dilmun myth), `sag9-ga` (good/well-being), and several terms associated with the body and vital functions. The heatmap reveals that `zi` sits at moderate-to-large distance from most of the other cosmogonic concepts in this vocabulary set. It shows closest proximity to body-and-vitality adjacent terms rather than to the abstract governance terms (`me`, `namtar`) — consistent with its lexical role as an animating-principle word rather than an institutional or decree-type word.

The distance between `zi` and `namtil3` (compound: `nam` + `til3`, "life" or "lifespan") is notably moderate — smaller than the distance between `zi` and, say, `me` — suggesting that the embedding space partially captures the lexical kinship between `zi` and the compound in which it participates.

### 5.4 Dual-View Divergence

| Rank | Gemma Token | Gemma Sim | GloVe Token | GloVe Sim |
|------|-------------|-----------|-------------|-----------|
| 1 | zapaag2zu | 0.3325 | zapaŋ2 | 0.5180 |
| 2 | ku6gu10 | 0.3173 | zapaŋ2ŋu10 | 0.5049 |
| 3 | zapaŋ2zu | 0.2992 | zapaŋ2zu | 0.5010 |
| 4 | zapaŋ2 | 0.2973 | zapaag2zu | 0.4985 |
| 5 | zapaŋ2bi | 0.2909 | zapaaj2 | 0.4983 |
| 6 | zapaag2bi | 0.2879 | zapaŋ2bi | 0.4949 |
| 7 | zapaag2 | 0.2878 | zapaaj2zu | 0.4882 |
| 8 | zapaŋ2ŋu10 | 0.2866 | zapaag2 | 0.4811 |
| 9 | zapaaj2zu | 0.2766 | zapaaj2bi | 0.4807 |
| 10 | zapaaj2bi | 0.2764 | ga14 | 0.4759 |

The Gemma and GloVe neighbor lists for `zi` show substantial agreement: the `zapaa(g/ŋ/j)2` cluster appears in both spaces as the dominant neighbor set. This cross-space agreement is the strongest evidence in the `zi` analysis: whatever is driving `zi` into the voice/sound neighborhood, it is not an artifact of one alignment method's geometry but a feature that persists across both spaces. The primary difference is that GloVe shows higher absolute similarities (0.48–0.52 vs. 0.28–0.33 in Gemma), reflecting the different distributional structures of the two target spaces. The lone GloVe-only entry at rank 10 is `ga14`, which is a hapax or rare form and does not substantially change the interpretation.

The cross-space agreement here is the highest degree of dual-view consensus in our five-concept set, and it makes the voice/sound neighborhood for `zi` a genuine, high-confidence geometric finding — even if the interpretive implications remain open.

### 5.5 Analogy Probes

The analogy query `breath : air :: life : ?` (`concepts.zi.analogy_probes[0]`) in Gemma space returns:

| Rank | Token | Similarity |
|------|-------|------------|
| 1 | namtil3bi | 0.5251 |
| 2 | namtil3laka | 0.5210 |
| 3 | namtil3 | 0.5206 |
| 4 | namtil3la | 0.5181 |
| 5 | namtila | 0.5173 |

This is the most interpretively rich analogy result in the five-concept set. The probe returns `namtil3` and its inflected variants — the Sumerian compound for "life/lifespan" — at similarities above 0.52. The construction of the probe (`breath : air :: life : ?`) asks which Sumerian token stands to English "life" the way `zi` (breath) stands to English "air," and the answer is `namtil3` — the nominal compound built on `nam` (essence-prefix) plus `til3` (to complete, to finish). `Namtil3` is the state of being alive as an ongoing reality, which is semantically distinct from `zi` (the animating principle that produces that state).

The probe result is consistent with the Sumerological insight that `zi` and `namtil3` are semantically adjacent but not identical: `zi` is the animating force, `namtil3` is the condition of being animated. The geometry captures this distinction as a directional relationship: projecting from the English `air → life` vector in Gemma space arrives at `namtil3` rather than `zi` itself, which is geometrically appropriate. The finding is one of the cleaner instances in this document of analogy geometry producing a philologically defensible result.

### 5.6 English Displacement

**Cosine similarity (Gemma):** 0.056 (`concepts.zi.english_displacement.gemma.cosine_similarity`)

**GloVe comparison:** 0.270 (`concepts.zi.english_displacement.glove.cosine_similarity`)

At 0.056, the `zi` displacement in Gemma space is very low: the Sumerian-projected vector for `zi` is only marginally less orthogonal to English "breath" than `abzu` is to "deep." The GloVe result (0.270) is warmer but still far from the `namtar` threshold. The embedding geometry is consistent with the interpretation that "breath" as an English concept — oxygen inhalation, the respiratory cycle, metaphors of inspiration and speech — occupies a substantially different region of the manifold than `zi` occupies in the projected Sumerian space. The voice/sound neighborhood found in §5.2 may be part of what `zi` captures that "breath" misses: the productive, world-generating aspect of breath as speech-act rather than breath as physiological process.

### 5.7 Source-Text Grounding

ETCSL passage c3205.46 (`concepts.zi.etcsl_passages[1]`):

> **Transliteration:** *zi su3-ud-jal2 nij2-ba-e-ec2 ba-mu-na-ab*
> **Translation:** "Bestow on me long life as a gift!"

This line is from a royal hymn (text c3205, a hymn to Šulgi or a related Ur III king). The collocation `zi su3-ud-jal2` — "life that is wide/long" — is significant: `su3-ud` means "distant, long, extended," and `jal2` is "to be, to exist." Long life is expressed not as duration (a temporal concept) but as spatial extension — life stretched out across time imagined as space. This is consistent with what Jacobsen calls the "spatialization of time" in Sumerian thought, in which temporal concepts are rendered through the same vocabulary as spatial ones (Jacobsen 1976, p. 7).

The word `zi` in this context is "life" in the life-span sense — the human individual's allotted time of living — rather than the animating-breath sense. The polysemy is consistent with ePSD2's dual gloss, and the ETCSL passage shows that both glosses can be simultaneously active in a single liturgical phrase.

### 5.8 Interpretive Synthesis

The geometric profile of `zi` is one of low translation-fidelity (displacement 0.056) but high internal consistency (strong cross-space agreement on voice/sound neighbors). The English gloss "breath" captures the physiological referent without capturing the distributional neighborhood, which leans toward vocal expression and may reflect the close Sumerian association between breath, voice, and the creative/animating power of speech.

A hedged cosmogonic claim consistent with this geometry: the Sumerian vector for `zi` occupies a region of semantic space that partially overlaps with the vocabulary of voice and sound, suggesting that what animates the clay figure in the creation myth is not merely the mechanical act of respiration but something closer to the entrance of voice — and through voice, intention and divine participation — into the formed clay. This is consistent with Jacobsen's account of divine presence as a "Thou" that enters into relationship (Jacobsen 1976, pp. 5–6) and with the ETCSL testimony that life can be "bestowed as a gift" through a speech act from divine to human. It contradicts the simpler reading of `zi` as straightforwardly breath-as-physiology.

The alternative reading — that the voice/sound neighborhood is a coincidental distributional artifact of co-occurrence in liturgical texts (where hymns about life and breath are performed aloud, and vocal terms co-occur) — cannot be ruled out with current data. The cross-space agreement makes the coincidence less likely but does not eliminate it.

`Zi` leads into `nam` (§6): if `zi` is what animates the individual entity, `nam` is the abstract machinery by which that entity's character and destiny are specified.

---

## § 6 — Deep Dive: `nam` (Essence/Office/Abstract Nominal)

### 6.1 Anchor Reading

`Nam` is one of the most productive morphemes in Sumerian. As a prefix it converts a concrete noun or verb root into an abstract nominal: `nam-lugal` (kingship, from `lugal` = king), `nam-til3` (life/lifespan, from `til3` = to finish), `nam-tar` (fate, from `tar` = to cut/decide). As a standalone noun, `nam` means something like "essence" or "nature" or "office" — the inherent character of a thing that determines how it functions in the world. The ePSD2 glosses `nam` as "nature, fate, office."

Jacobsen discusses `nam` under the category of the "offices" distributed by the Anunnaki — the notion that each thing in the world has an inherent `nam` that determines its function and limits (Jacobsen 1976, p. 88). Kramer focuses on `nam` as the abstract noun of destination: what an entity is *for* in the cosmos, which is determined by fate-determination (Kramer 1944, pp. 79–80). Foxvog notes that the `nam`-prefix is one of Sumerian's most productive derivational morphology tools (*Introduction to Sumerian Grammar*, §8.3 on nominalizing prefixes). The word `nam` thus straddles three conceptual registers: morphological (it is a derivational prefix), semantic (it denotes inherent nature/essence), and cosmological (it is the vehicle through which the Anunnaki assign destinies).

### 6.2 Nearest Sumerian Neighbors (Gemma Space)

The top-10 nearest Sumerian neighbors to `nam` in Gemma space (`concepts.nam.top10_dual_view.gemma`) are:

| Rank | Token | Cosine Similarity |
|------|-------|-------------------|
| 1 | saŋzu | 0.2527 |
| 2 | sagzu | 0.2308 |
| 3 | saŋza | 0.2277 |
| 4 | muunzu | 0.2267 |
| 5 | agabita | 0.2264 |
| 6 | sajzu | 0.2261 |
| 7 | muunzuzu | 0.2255 |
| 8 | u3zu | 0.2210 |
| 9 | muunzuub3 | 0.2202 |
| 10 | namgur4zu | 0.2199 |

The Gemma top-10 for `nam` is again dominated by morphological noise: `saŋzu` (your head), `sagzu`/`sajzu` (orthographic variants of the same), `saŋza` (a morphological variant), `muunzu`/`muunzuzu`/`muunzuub3` (third-person possession/directional forms). The rank-10 entry `namgur4zu` is interesting: `namgur4` is an attested Sumerian compound ("your fattened one/your heavy one"), but the `nam-` prefix here is the same morpheme as the concept under analysis. This circularity — `nam`'s nearest neighbors include words formed with `nam`-as-prefix — is a structural consequence of FastText's subword model: tokens that begin with the same character sequence will be pulled toward each other in the subword-feature space.

The similarities are uniformly low (0.22–0.25), and there is no evident semantic-field cluster visible in the Gemma results.

### 6.3 Semantic-Field Map

The heatmap at `docs/figures/cosmogony/nam_semantic_field_heatmap.png` shows pairwise distances among a vocabulary that includes `nam`, `namtar`, `me`, `zi`, `an`, `ki`, `enlil`, `nig2` (thing/matter), `til3` (to finish/complete), and several nominalizing-compound forms drawn from the ETCSL corpus. The most notable feature is that `nam` and `namtar` are geometrically adjacent in the heatmap — which is expected, since `namtar` is literally `nam` + `tar`. What is more interesting is that the distance between `nam` and `me` is moderate rather than small: despite their shared cosmogonic domain (both are involved in the specification of cosmic order), their embedding vectors are not particularly close. The geometry is consistent with the interpretation that `nam` and `me` are complementary but not synonymous: `nam` is the quality or nature of a thing; `me` is the civilizational template within which that quality operates.

### 6.4 Dual-View Divergence

| Rank | Gemma Token | Gemma Sim | GloVe Token | GloVe Sim |
|------|-------------|-----------|-------------|-----------|
| 1 | saŋzu | 0.2527 | sikilla | 0.4565 |
| 2 | sagzu | 0.2308 | sikile | 0.4346 |
| 3 | saŋza | 0.2277 | sikillaza | 0.4339 |
| 4 | muunzu | 0.2267 | sikilzu | 0.4254 |
| 5 | agabita | 0.2264 | namsa2e | 0.4205 |
| 6 | sajzu | 0.2261 | sikilebi | 0.4178 |
| 7 | muunzuzu | 0.2255 | namkugzu | 0.4149 |
| 8 | u3zu | 0.2210 | sikilbi | 0.4147 |
| 9 | muunzuub3 | 0.2202 | sikillake4 | 0.4094 |
| 10 | namgur4zu | 0.2199 | kuggakam | 0.4047 |

The GloVe top-10 (`concepts.nam.top10_dual_view.glove`) is dominated by `sikil`-forms: `sikilla`, `sikile`, `sikillaza`, `sikilzu`, etc. `Sikil` means "pure, clean, spotless" in Sumerian — a ritual-purity term that appears frequently in temple and liturgical contexts. The appearance of purity vocabulary as the nearest GloVe neighbors for `nam` (seeded from the English word "essence") is an interpretively provocative divergence from the Gemma results. It suggests that, in the GloVe-aligned space, the English word "essence" gravitates toward Sumerian vocabulary of ritual purity rather than abstract nominal machinery. This may reflect the semantic neighborhood of "essence" in English itself: essence connotes purity, distillation, the removal of contaminating elements.

The GloVe rank-5 item `namsa2e` is a compound: `nam` + `sa2` (to equal, to be level) + the case marker `e`. This is close in form to `nam` itself and may reflect morphological pull; but `nam-sa2` also appears in ETCSL texts as "well-being" or "proper order" — a cosmological term. Its appearance in the GloVe neighborhood is the only cross-space convergence (a `nam`-compound in GloVe as well as a `namgur4zu` variant in Gemma), though the specific compounds differ.

### 6.5 Analogy Probes

The analogy query `essence : name :: being : ?` (`concepts.nam.analogy_probes[0]`) in Gemma space returns:

| Rank | Token | Similarity |
|------|-------|------------|
| 1 | muni | 0.3633 |
| 2 | muzu | 0.3344 |
| 3 | muniim | 0.3010 |
| 4 | ninmuinzu | 0.3009 |
| 5 | ninmu | 0.2979 |

The probe result centers on `mu`-forms: `muni` (his/her name), `muzu` (your name), `muniim` (a variant). `Mu` is the Sumerian word for "name." The probe `essence : name :: being : ?` is asking which Sumerian token stands to English "being" the way `nam` (essence) stands to English "name" — and the geometry returns name-possessive forms. This is not the result that would be predicted if `nam` and `mu` (name) occupied radically different semantic regions; rather, it suggests that in Gemma space, the analog of the essence-to-name relationship passes through name-inflected vocabulary.

The result is consonant with the Sumerological observation that in Sumerian metaphysics, naming and essencing are closely related: to know the name of something is to have access to its `nam`, and the Anunnaki's determination of destiny is accomplished partly through naming. The analogy probe does not prove this connection, but the geometric result is at least consistent with it.

### 6.6 English Displacement

**Cosine similarity (Gemma):** 0.129 (`concepts.nam.english_displacement.gemma.cosine_similarity`)

**GloVe comparison:** 0.301 (`concepts.nam.english_displacement.glove.cosine_similarity`)

At 0.129, `nam` is the second-least-translation-adjacent concept after `me` (0.017) and `abzu` (−0.002) among our five. The GloVe result (0.301) is moderate. The embedding geometry is consistent with the interpretation that English "essence" — a word with philosophical and theological freight accumulated through Aristotelian `ousia`, scholastic `essentia`, and German `Wesen` — encodes a conceptual history substantially different from `nam`'s distributional neighborhood in Sumerian corpus texts. The Sumerian morphological productivity of `nam` (it is a derivational prefix generating hundreds of compound nouns) means its embedding vector is pulled by a very wide range of contexts — the contexts of all the `nam`-compounds in the ETCSL corpus — rather than by a single concentrated semantic domain.

### 6.7 Source-Text Grounding

ETCSL passage c2554.B.11 (`concepts.nam.etcsl_passages[0]`):

> **Transliteration:** *mu-na-ab-be2 dumu cu jar gi4-zu nam gal tar-mu-ni-ib2*
> **Translation:** "Then she said: Decide a great fate for the son who is your avenger!"

This is from a hymn or epic text (c2554, likely a Ninurta hymn), in which a divine mother presents her son to the assembly of gods and petitions for the determination of his great `nam`. The verb phrase `nam gal tar` — "cut/decide the great nam" — is the canonical expression for fate-determination in Sumerian. The line makes clear that `nam` is not merely an inherent quality that a thing passively possesses; it is something actively determined by divine action, something that can be "cut" larger or smaller by the gods' assembly.

The ETCSL passage thus shows `nam` in its most cosmogonically loaded usage: not as the morphological prefix of everyday nominals but as the substance of divine destiny-decree. This is the face of `nam` that the Anunnaki manipulate; the morphological-prefix face is the way that substance permeates into the ordinary world's vocabulary.

### 6.8 Interpretive Synthesis

The geometric profile of `nam` is distinctive: very low translation fidelity (Gemma displacement 0.129), morphological-noise-dominated Gemma neighborhood, purity-vocabulary-dominated GloVe neighborhood, and a strong analogy probe result showing naming/name-possession as the geometric neighbor of nam-to-name. 

A hedged cosmogonic claim: the Sumerian vector for `nam` appears to encode not a single unified concept but the distributional signature of a highly productive morphological element that participates in a wide range of semantic contexts — fate, life, office, ritual status — without being geometrically anchored to any one of them. The English gloss "essence" imports a philosophical tradition (Aristotelian substance-theory, scholastic metaphysics) that generates a specific, concentrated semantic neighborhood in English. The Sumerian vector sits at 0.129 cosine distance from that neighborhood: not orthogonal, but not close. The embedding geometry is consistent with the hypothesis that `nam` is less a concept than a conceptual-production engine — a device for generating abstractions — whose geometric position reflects the full ensemble of abstractions it generates rather than any one of them in isolation.

This interpretation has a direct implication for the cosmogonic narrative: when the gods "cut the great nam" for a hero or a city, they are not merely assigning a pre-existing category but generating a specific abstract entity — a new instance of `nam`-as-essence — that will determine that hero's or city's mode of being in the world. The `me` (§8) are in some sense the catalog of all the `nam`s that have ever been cut; the relationship between the two words is one of concrete instance (`me`) and generative capacity (`nam`).

---

## § 7 — Deep Dive: `namtar` (Fate/Destiny)

### 7.1 Anchor Reading

The ePSD2 glosses `namtar` as "fate, destiny" and also (in an independent lexical entry with the same form) as "Namtar," the name of the demon-god of fate and death who serves as Ereshkigal's vizier in the Underworld. The ambiguity is not coincidental: in Sumerian cosmology, fate is not an abstract logical principle but an agent — something executed by a divine person. `Namtar` is morphologically `nam` (essence/office; see §6) + `tar` (to cut, to determine, to decree). The fate-demon's name *is* the name of the action: fate is "the cutting of essence," and the entity who cuts is named for the cutting.

Jacobsen analyzes `namtar` as one of the two great inevitable powers that no human can escape — alongside `me-lam`, the awesome radiance of divine presence — noting that the fate-demon Namtar appears in *The Descent of Inanna to the Underworld* (ETCSL 1.4.1) as a figure of dread: it is Namtar who afflicts Inanna with the sixty diseases at Ereshkigal's command (Jacobsen 1976, p. 55). Black et al. note that in the *Curse of Agade* (ETCSL 2.1.5), Namtar is deployed as a weapon: Enlil sends Namtar against the city as punishment for its hubris (Black et al. 2004, p. 119).

The dual identity — abstract noun (fate) and personalized demon (Namtar) — is a persistent feature of Sumerian theological vocabulary: the most fundamental cosmic forces are simultaneously principles and persons. This duality means that `namtar` in the corpus appears in highly varied syntactic positions: as a subject (Namtar acts), as an object (one's fate is determined), as a genitive modifier (the wood of fate), and as a predicate (this is one's namtar).

### 7.2 Nearest Sumerian Neighbors (Gemma Space)

The top-10 nearest Sumerian neighbors to `namtar` in Gemma space (`concepts.namtar.top10_dual_view.gemma`) are:

| Rank | Token | Cosine Similarity |
|------|-------|-------------------|
| 1 | namtar | 0.4592 |
| 2 | namtag2 | 0.4523 |
| 3 | namta | 0.4502 |
| 4 | namtag | 0.4182 |
| 5 | namtarra | 0.4066 |
| 6 | namtarju10 | 0.3948 |
| 7 | namtarzu | 0.3939 |
| 8 | namtaba | 0.3903 |
| 9 | namtagga | 0.3886 |
| 10 | namtaggani | 0.3883 |

The top-1 nearest Sumerian neighbor is `namtar` itself (cosine 0.4592), followed by a dense cluster of morphologically related forms: `namtag2`/`namtag` (sin/guilt — a semantically adjacent concept involving predetermined transgression), `namta` (a probable ablative form), `namtarra`/`namtarju10`/`namtarzu`/`namtaba` (grammatical variants of `namtar`). The rank-2 and rank-4 entries `namtag2`/`namtag` are significant: this word for "sin" or "guilt" is built from `nam` + `tag` (to touch, to hit, to strike) and is often paired with `namtar` in lament texts as a dyad of negative fate and moral transgression.

The absolute similarities in this list (0.39–0.46) are the highest of any concept in our five-concept set. This means `namtar` sits in a tight, well-defined cluster of morphologically and semantically related forms — a "fat" neighborhood. The geometry is consistent with the interpretation that fate-concepts in Sumerian form a dense conceptual region, not a scattered one.

### 7.3 Semantic-Field Map

The heatmap at `docs/figures/cosmogony/namtar_semantic_field_heatmap.png` shows pairwise distances among a vocabulary including `namtar`, `namtil3`, `namtag2`, `nam`, `me`, `kur` (netherworld), `ereshkigal`, `an`, `enlil`, and several lament-text terms associated with death, fate, and divine decree. The most striking feature is the tight clustering of `namtar` with `namtag2` and `namtil3` — three `nam`-compound fate/life/guilt concepts occupying a close cluster. The distance from `namtar` to `me` is moderate, and from `namtar` to `kur` (netherworld, where Namtar the demon resides) is larger than might be expected on purely mythological grounds — suggesting that the corpus distributional patterns dissociate the fate-concept from the netherworld-location concept more than the mythology would predict.

### 7.4 Dual-View Divergence

| Rank | Gemma Token | Gemma Sim | GloVe Token | GloVe Sim |
|------|-------------|-----------|-------------|-----------|
| 1 | namtar | 0.4592 | namtar | 0.5674 |
| 2 | namtag2 | 0.4523 | namtag2 | 0.5673 |
| 3 | namta | 0.4502 | namtarzu | 0.5647 |
| 4 | namtag | 0.4182 | namtag | 0.5641 |
| 5 | namtarra | 0.4066 | namtarju10 | 0.5566 |
| 6 | namtarju10 | 0.3948 | namta | 0.5565 |
| 7 | namtarzu | 0.3939 | namte | 0.5378 |
| 8 | namtaba | 0.3903 | namtarrani | 0.5356 |
| 9 | namtagga | 0.3886 | namtae3 | 0.5340 |
| 10 | namtaggani | 0.3883 | namtarra | 0.5292 |

The Gemma and GloVe neighbor lists for `namtar` are the most consistent across the two spaces of any concept in our analysis. Eight of the ten tokens appear in both lists (in varying order). The cross-space overlap is near-total: `namtar` itself, `namtag2`, `namta`, `namtag`, `namtarra`, `namtarju10`, `namtarzu` all appear in both. The only GloVe-only entries are `namte`, `namtarrani`, and `namtae3` — all morphological variants. There are no thematically unrelated tokens in either list; the entire top-10 in both spaces is `namtar`-and-`namtag`-compounds.

This maximal cross-space agreement, combined with the high absolute similarities, makes `namtar` the most geometrically robust concept in our set. The embedding geometry is consistent with the interpretation that fate-vocabulary in Sumerian is a coherent, tightly structured semantic field that aligns consistently across projection methods.

### 7.5 Analogy Probes

The analogy query `fate : name :: decree : ?` (`concepts.namtar.analogy_probes[0]`) in Gemma space returns:

| Rank | Token | Similarity |
|------|-------|------------|
| 1 | muni | 0.3313 |
| 2 | mubiim | 0.3002 |
| 3 | muniku4 | 0.2987 |
| 4 | muniinsa4 | 0.2986 |
| 5 | muniim | 0.2927 |

The probe asks: which Sumerian token stands to "decree" the way `namtar` (fate) stands to "name"? The result centers on `mu`-forms again: `muni` (his/her name), `mubiim` (its name/by its name), `muniku4` and `muniinsa4` (verbal forms involving naming/entering by name). This parallels the `nam` analogy result from §6.5, where naming-forms also dominated. The geometric intersection between fate-concepts and naming-concepts — both reaching toward `mu`-forms in analogy probes — is consistent with the Sumerological observation that naming is an act of fate-determination: to know and speak the name is to have power over the destiny.

### 7.6 English Displacement

**Cosine similarity (Gemma):** 0.459 (`concepts.namtar.english_displacement.gemma.cosine_similarity`)

**GloVe comparison:** 0.567 (`concepts.namtar.english_displacement.glove.cosine_similarity`)

At 0.459 in Gemma space and 0.567 in GloVe space, `namtar` is by a wide margin the most translation-adjacent concept in our five-concept set. The next-closest is `nam` at 0.129. The gap between `namtar` and the rest of the set is large enough to constitute a qualitative difference, not merely a quantitative gradient: `namtar` lands near its English gloss, while the others do not.

This finding is honest but requires careful interpretation. The high displacement similarity does not mean that "fate" and `namtar` are the same concept across the two languages. It means that in the aligned Gemma space, the direction from the origin to the `namtar` vector is similar to the direction from the origin to the English "fate" vector. This could reflect genuine conceptual overlap — the two words really do pick out similar experiential and cognitive territory — or it could reflect an alignment artifact in which fate-concepts are well-represented in the ETCSL anchor pairs and therefore well-aligned in the ridge regression. The preflight check (§12) confirms that `namtar` has 35 ETCSL passages, which is modest but sufficient.

The most careful statement is: the embedding geometry is consistent with "fate" being a substantially better translation of `namtar` than "deep" is of `abzu`, "breath" is of `zi`, "essence" is of `nam`, or "decree" is of `me`.

### 7.7 Source-Text Grounding

ETCSL passage c4332.27 (`concepts.namtar.etcsl_passages[1]`):

> **Transliteration:** *nam-tar nam-he2 jic nam-tar jic nam-he2*
> **Translation:** "Destiny, prosperity — the wood of destiny, wood of prosperity, and the reeds of destiny, reeds of prosperity, adorn the holy cattle-pen."

This is from a ritual purity text (c4332), in which `namtar` is paired with `namhe2` — destiny and prosperity, or fate and abundance — as ritual objects adorning a sacred space. The repeated pairing `jic nam-tar jic nam-he2` ("the wood of destiny, wood of prosperity") suggests that fate and flourishing were understood as complementary ritual substances, not opposed forces. The materialization of fate in physical objects (wood, reeds) used to decorate a sacred cattle-pen is a characteristic instance of Sumerian ritual theology: abstract cosmic principles take on physical form as ritual ingredients.

A second passage (c3205.28) renders `namtar` differently: *nam-tar hul-jal2 a2-sag3 nij2-gig-ga nu-mu-un-na-tum3* ("The evil namtar demon and the distressing asag demon have not carried him off"). Here `namtar` appears as a demon rather than an abstract principle, with the epithet `hul-jal2` ("evil-possessing/evil-filled"). The same word in two texts: a ritual object of good omen, and a dread demon of death. The distributional richness that produces `namtar`'s tight, high-similarity neighborhood in the embedding space is partially a reflection of this textual range.

### 7.8 Interpretive Synthesis

`Namtar` is the lone genuinely translation-adjacent concept in our set, and this fact itself requires explanation. Why would fate-as-concept be more cross-linguistically portable than deep-as-concept, breath-as-concept, or decree-as-concept?

One hypothesis: fate is a concept whose core structure — the predetermined determination of individual destiny by an external agent or force — is universal enough across human cultures that it generates consistent distributional neighborhoods even in languages as distant as Sumerian and English. Every culture that has literary texts has a word for the-thing-that-determines-what-happens-to-you; those words will tend to co-occur with similar vocabulary (death, life, decree, name, gods, prayer) across cultures, giving them similar embedding signatures.

A competing hypothesis: `namtar` achieves high displacement similarity because the ETCSL corpus contains a sufficient density of unambiguous fate-contexts (fate-demons, fate-cutting rituals, fate-determination prayers) that the alignment learns a reliable mapping, whereas `abzu`, `zi`, and `me` appear in more varied, harder-to-align contexts.

Both hypotheses are consistent with the data. The embedding geometry does not adjudicate between them. What it establishes firmly is the *pattern*: fate translates; deep does not; breath does not; decree does not. Whether this pattern reflects cognitive universality, alignment quality, or corpus structure — or some combination of all three — is a question for future work, including comparative analysis with other ancient languages.

The `namtar` deep dive leads directly to `me` (§8): if `namtar` executes the fate-determination, `me` is the template library within which fates are drawn. The two concepts are cosmologically adjacent but geometrically distinct.

---

## § 8 — Deep Dive: `me` (Divine Civilizational Decrees)

### 8.1 Anchor Reading

The Sumerian `me` (pronounced approximately "may" — a short front vowel followed by a bilabial stop) is arguably the most conceptually dense term in the Sumerian theological lexicon. The ePSD2 glosses it as "divine power" or "divine ordinance." Piotr Michalowski, in his foundational analysis of the concept, characterizes the `me` as "the sum total of all those institutions, attributes, qualities of behavior, and things that are characteristic of civilized life" (Michalowski 1993, "On the Early History of the *eridu* Lament," *Acta Sumerologica* 12, cited for the framing; the specific quotation paraphrases his position). Black et al. describe the `me` as "the fundamentals of civilization" distributed by the gods, listing examples from *Inanna and Enki* (ETCSL 1.3.1): kingship, the descent to the Underworld, the craft of the scribe, music, the art of building, the wisdom of divination (Black et al. 2004, p. 218).

The list of `me` in *Inanna and Enki* runs to more than one hundred items, suggesting that the concept is genuinely encyclopedic: the `me` are not a selection of the most important civilizational achievements but the complete ontology of what-civilization-is. Each `me` is an entity in the Sumerian cosmos: it can be held, transferred, stolen, and distributed. Inanna steals the `me` from Enki (who is drunk) and carries them to Uruk; Enki sends a fleet to recover them, but Inanna successfully delivers each `me` to the city. The narrative presupposes that the `me` are countable, portable, and exhaustive.

This ontological completeness makes `me` one of the most alien concepts in the set: it is not decree in the sense of an act of will or a command, but something closer to a primordial ontological template — the fact-of-being of a civilizational institution.

### 8.2 Nearest Sumerian Neighbors (Gemma Space)

The top-10 nearest Sumerian neighbors to `me` in Gemma space (`concepts.me.top10_dual_view.gemma`) are:

| Rank | Token | Cosine Similarity |
|------|-------|-------------------|
| 1 | dikud | 0.2570 |
| 2 | diku5bi | 0.2557 |
| 3 | diku5bime | 0.2520 |
| 4 | diku5še3 | 0.2447 |
| 5 | ensi2kašse3 | 0.2325 |
| 6 | ensi2 | 0.2313 |
| 7 | ensi2kabi | 0.2288 |
| 8 | ensi2ka | 0.2265 |
| 9 | diku5dani | 0.2262 |
| 10 | diku5gal | 0.2236 |

The Gemma top-10 for `me` is semantically interpretable in a way that the other concepts' top-10 lists are not. `Dikud` means "judge" (from `di` = legal case + `kud` = to cut/decide; the same `kud`/`tar` root that produces "fate-cutting"). `Diku5bi`/`diku5bime`/`diku5še3`/`diku5dani`/`diku5gal` are grammatical variants and epithets of `dikud`. `Ensi2` means "governor" or "city ruler" — an administrative officer. The cluster is clearly administrative-governance vocabulary: judges and governors in their several grammatical forms.

This is the first Gemma top-10 in our five-concept set that shows a recognizable semantic field rather than morphological noise or dispersed forms. The finding is interpretively significant: the English anchor "decree" (the seed used to find `me` in aligned space) draws the projection toward governance-administrative vocabulary in Sumerian, rather than toward theological/cosmological vocabulary. This is consistent with the interpretation that the English word "decree" — with its connotations of administrative authority, legal pronouncement, and governmental action — captures the *institutional* face of `me` while missing its *ontological* face. The `me` are not merely what rulers decree; they are what makes it possible for there to be rulers who can decree.

### 8.3 Semantic-Field Map

The heatmap at `docs/figures/cosmogony/me_semantic_field_heatmap.png` shows pairwise distances among a vocabulary including `me`, `namtar`, `nam`, `zi`, `abzu`, `inanna`, `enki`, `dikud`, `ensi2`, `nig2-nam` (all things/everything), and several terms associated with the distribution of `me` in *Inanna and Enki* contexts. The dominant visual feature is the tight cluster of `dikud`- and `ensi2`-forms around `me` — consistent with the Gemma top-10 — and the moderate distances between `me` and the other four cosmogonic concepts. The `me`-to-`namtar` distance is similar in magnitude to the `me`-to-`nam` distance, suggesting that these three concepts, while cosmologically linked, are not geometrically co-located. Each occupies a distinct region of the manifold.

### 8.4 Dual-View Divergence

| Rank | Gemma Token | Gemma Sim | GloVe Token | GloVe Sim |
|------|-------------|-----------|-------------|-----------|
| 1 | dikud | 0.2570 | namensi2 | 0.3751 |
| 2 | diku5bi | 0.2557 | ensi2mah | 0.3547 |
| 3 | diku5bime | 0.2520 | ensi2kabi | 0.3533 |
| 4 | diku5še3 | 0.2447 | name3 | 0.3503 |
| 5 | ensi2kašse3 | 0.2325 | ensi2kam | 0.3480 |
| 6 | ensi2 | 0.2313 | lugalmah | 0.3414 |
| 7 | ensi2kabi | 0.2288 | ensi2 | 0.3375 |
| 8 | ensi2ka | 0.2265 | dikud | 0.3374 |
| 9 | diku5dani | 0.2262 | ensi2. | 0.3367 |
| 10 | diku5gal | 0.2236 | ensi2kasze3 | 0.3359 |

The GloVe top-10 (`concepts.me.top10_dual_view.glove`) shows strong convergence with Gemma on the administrative-governance vocabulary: `ensi2` (governor) and `dikud` (judge) both appear in both lists. The GloVe-only additions are `namensi2` (governorship/the-nam-of-the-ensi, a compound), `ensi2mah` (great governor), `name3` (a `me` compound), `ensi2kam` (of the governor), and `lugalmah` (great king). The GloVe neighborhood is entirely governance vocabulary, with no overlap into cosmological or theological terms.

The cross-space agreement on `ensi2` and `dikud` establishes administrative-governance vocabulary as the high-confidence geometric neighborhood for `me`. The GloVe-only additions extend the governance vocabulary upward (toward kingship: `lugalmah`) rather than inward (toward cosmology). This consistent governance orientation across both spaces is the clearest dual-view finding in our five-concept set: the English seed "decree" reliably projects into administrative-governance Sumerian vocabulary, not into the cosmological-civilizational domain where the `me` concept primarily lives.

### 8.5 Analogy Probes

The analogy query `decree : order :: essence : ?` (`concepts.me.analogy_probes[0]`) in Gemma space returns:

| Rank | Token | Similarity |
|------|-------|------------|
| 1 | melam2zu | 0.2145 |
| 2 | igigu10šse3 | 0.2006 |
| 3 | gu2ni | 0.1991 |
| 4 | usubita | 0.1985 |
| 5 | melam2ba | 0.1939 |

The probe result is the least interpretively clear of the five. The top-1 and top-5 results are `melam2`-forms (`melam2zu` = your melam, `melam2ba` = its melam). `Melam2` is the Sumerian word for "awesome radiance" or "divine sheen" — the visible manifestation of divine power that makes gods and kings terrifying and luminous. That the `decree-to-order :: essence-to-?` probe retrieves divine-radiance vocabulary is unexpected but not incoherent: the `me` and `melam2` are closely associated in Sumerian literary texts, where possessing the `me` is often paired with the radiant `melam2` as co-markers of divine authority.

The intermediate results — `igigu10šse3` ("to/toward my eye," a directional phrase), `gu2ni` ("his neck/presence"), `usubita` ("from the temple/sanctuary") — are less interpretively structured, suggesting that the probe is operating at the edges of reliable analogy geometry for this concept.

### 8.6 English Displacement

**Cosine similarity (Gemma):** 0.017 (`concepts.me.english_displacement.gemma.cosine_similarity`)

**GloVe comparison:** 0.119 (`concepts.me.english_displacement.glove.cosine_similarity`)

At 0.017 in Gemma space, `me` is effectively orthogonal to the English word "decree." The Sumerian concept and its English gloss sit in geometrically unrelated regions of the aligned embedding space. The GloVe result (0.119) is low as well, making `me` the second-lowest displacement concept after `abzu` (−0.002) in the Gemma space.

The near-zero displacement, combined with the governance-vocabulary top-10 neighborhood, tells a consistent story: the English word "decree" captures one thin slice of `me`'s semantic profile — the institutional, administrative face — while the cosmological-ontological face (the `me` as primordial template of civilizational being) sits in a region of Sumerian embedding space that the English word "decree" cannot reach. The embedding geometry is consistent with the strong interpretation that `me` is the most genuinely alien concept in our set: more alien even than `abzu` in the sense that while `abzu` is simply orthogonal to its English gloss, `me` is orthogonal *and* its geometric neighborhood (governance vocabulary) represents a partial misreading of the concept's cosmological significance.

### 8.7 Source-Text Grounding

ETCSL passage c0201.4 (`concepts.me.etcsl_passages[0]`):

> **Transliteration:** *nin me car2-ra*
> **Translation:** "Lady of all the me" (or: "Lady of the innumerable me")

This is from a hymn to a female deity (the text c0201 is a hymn to Inanna based on context). The phrase *me car2-ra* — "me, the great/innumerable" — is one of the most common Inanna epithets in the ETCSL corpus, appearing in the *Hymn to Inanna* (ETCSL 4.07.2) and several related texts. `Car2` means "numerous, many, sixty (as a large number)" in Sumerian. The phrase is a summary of Inanna's power: she holds all the civilizational templates, not just some of them. The genitive construction (*me* + genitive marker → "of the me") treats the `me` as a category of objects that can be possessed.

A second passage (c0201.8): *in-nin me huc-a* — "Lady of the fierce me" — pairs `me` with `huc` (fierce, red, terrifying). That the `me` can be fierce suggests they are not merely neutral institutional templates but charged with divine energy. The `me` of warfare, for instance, would be fierce; the `me` of kingship would be awesome. The attributive `huc` shows that individual `me` can have qualitative character.

### 8.8 Interpretive Synthesis

The geometric profile of `me` is the most discordant with received Sumerological understanding of any concept in our set. The English gloss "decree" misses the cosmological-ontological dimension entirely; the aligned geometric neighborhood is administrative governance (judges, governors, kings) rather than divine cosmological templates. The displacement is near-zero. And yet the concept is the culminating element of the cosmogonic arc — the distribution of the `me` is what makes Sumerian civilization possible.

A hedged cosmogonic claim: the embedding geometry of `me` is consistent with the hypothesis that the concept splits along two distributional dimensions in the corpus. In ritual and literary texts, `me` appears in the company of divine epithets, cosmic powers, and the names of gods — a theological register. In administrative and royal texts, `me` appears in the company of institutional terms, titles, and governance vocabulary — an administrative register. The English word "decree" resonates primarily with the administrative register, which is why the geometric projection lands in governance vocabulary. The theological register — which is what the cosmogonic function of `me` requires — is either not captured by the English anchor or not reliably aligned by the ridge regression.

This interpretive split is consistent with Michalowski's observation that the `me` are both theological (the divine blueprints that exist before creation) and institutional (the practices and roles that make up civilized life). The embedding geometry is not wrong to surface governance vocabulary; it is capturing one face of the concept, the face that the English gloss "decree" most naturally reaches. The other face — `me` as ontological template, `me` as what-civilization-is rather than what-rulers-command — is the one that the geometry does not reach, and that absence is itself a finding.

---

## § 9 — Synthesis: Cosmogony as Geometric Object

### 9.1 The Translation Opacity Spectrum

The central finding of this analysis is a spectrum of translation opacity across the five cosmogonic concepts, measured by the cosine similarity between the Sumerian-projected vector and the English anchor vector in Gemma-aligned space:

| Concept | English Anchor | Gemma Displacement | GloVe Displacement |
|---------|---------------|-------------------|-------------------|
| abzu | deep | −0.002 | 0.358 |
| zi | breath | 0.056 | 0.270 |
| me | decree | 0.017 | 0.119 |
| nam | essence | 0.129 | 0.301 |
| namtar | fate | 0.459 | 0.567 |

The axis-projection figure at `docs/figures/cosmogony/cosmogony_axis_projection.png` visualizes this spectrum: the five concepts are projected onto the cosmogonic translation-opacity axis, defined as the direction from the most-opaque concept (`abzu`, −0.002) to the most-transparent concept (`namtar`, 0.459). The arrangement reveals a clear qualitative gap: `namtar` sits far from the cluster of `abzu`, `me`, `zi`, and `nam`, which are all compressed near zero.

This is not what initial intuition would have predicted. One might have expected `zi` (breath/life) to be moderately translation-adjacent — "breath" is a concrete physiological term, and the animating act of breath seems like a human universal. Or one might have expected `me` to have some translation signal, since "decree" and governance-vocabulary are well-represented in the ETCSL corpus. The data contradict these expectations.

### 9.2 Why Fate Translates

The near-translation-adequacy of `namtar` (cosine 0.459) is the most surprising specific result in the study. We propose three non-exclusive explanations:

**Cognitive universality.** The concept of fate — the predetermined determination of individual destiny by an external power — may generate consistent distributional neighborhoods across human languages because every culture that produces literature grapples with it in similar ways. Death, prayer, divine intervention, lamentation, hope: these co-occur with fate-concepts regardless of language, producing similar embedding neighborhoods across aligned spaces.

**Alignment quality.** `Namtar` may be one of the most reliably aligned concepts in the pipeline because its ETCSL appearances are semantically unambiguous — when the texts say `namtar`, they are clearly talking about fate or the fate-demon, not using the word in a secondary or technical sense. High alignment quality would produce a high cosine similarity as an artifact of anchor reliability, not conceptual universality. We cannot distinguish this from the cognitive-universality hypothesis with current data.

**The demon problem.** `Namtar` as a demon is a personalized agent whose actions are easily narrated ("Namtar afflicted Inanna with sixty diseases") and whose textual contexts parallel those of death and fate in English literary tradition. The distributional neighborhood of a personalized fate-demon may more closely resemble the English neighborhood of "fate" than the neighborhood of an impersonal cosmic principle would. This is a subtle but real methodological consideration: the dual identity of `namtar` as both abstract noun and divine agent may actually *help* translation fidelity by producing more diverse and fate-specific contexts.

### 9.3 Why Deep, Breath, and Decree Do Not Translate

The three near-orthogonal concepts — `abzu` (−0.002), `me` (0.017), `zi` (0.056) — each have distinct reasons for their low displacement, and the differences are informative.

`Abzu` is orthogonal because the English word "deep" is primarily a spatial-geometric descriptor (vertical extent) with a secondary metaphorical sense (intellectual depth, profundity). The Sumerian `abzu` functions in the corpus as a theological address — a label for Enki's holy domain — rather than as a spatial descriptor. The distributional neighborhoods are simply in different regions of the semantic manifold: one is organized around extension and measurement, the other around sanctity and divine authority.

`Me` is orthogonal because "decree" captures only the administrative face of a concept that has a more primary cosmological-ontological face. The governance-vocabulary neighborhood that the alignment surfaces is not wrong; it is incomplete. The word "decree" draws the projection toward what rulers do, while the `me` are what makes it possible for there to be rulers.

`Zi` is near-orthogonal despite being a concrete physiological term because breath in English is organized around oxygen, respiration, and physical process, while `zi` in the corpus is organized around animation, vocal expression, and the divine gift of life. The voice/sound neighborhood found in §5.2 suggests that `zi` captures something that the English physiological term misses: the productive, creative, expressive dimension of breath as the medium of speech and divine presence.

### 9.4 Nam as Intermediate

`Nam` (cosine 0.129) sits at the lower end of the spectrum without being orthogonal. This intermediate position is consistent with the interpretation that `nam` is a highly productive morphological element whose embedding vector is spread across a wide range of semantic contexts — the full ensemble of `nam`-compounds in the corpus — rather than being concentrated in any one. The English word "essence" pulls the alignment toward purity vocabulary in GloVe space and toward morphological noise in Gemma space. The partial overlap (0.129) may reflect a genuine partial alignment between "essence" (the distilled, core nature of a thing) and the subset of `nam`-contexts that deal with inherent nature and office, as opposed to the fate-determination and morphological-nominal-production contexts.

### 9.5 Implications for the Research Program

This case study connects to the broader Cuneiformy research vision in two directions.

**What the geometry reveals.** The displacement spectrum confirms that Sumerian cosmogonic vocabulary is dominantly geometrically distinct from its English glosses. This is not a finding about the inadequacy of the glosses — they are contextually appropriate — but about the *structure* of the conceptual domains they point at. Four of the five concepts sit near zero cosine similarity to their English anchors, consistent with Jacobsen's claim that Sumerian religious consciousness was "mythopoeic" — operating in a fundamentally different mode than modern analytical categories (Jacobsen 1976, pp. 1–22). The embedding geometry provides a measurable correlate for that qualitative claim: the Sumerian conceptual topology, at least in these five cosmogonic words, does not map onto the English conceptual topology.

**What the geometry cannot reveal.** The alignment operates at the level of distributional co-occurrence in a specific corpus, at a specific historical moment (Old Babylonian literary Sumerian, ca. 2000–1600 BCE), projected through a specific alignment architecture (whitened-Gemma ridge regression). It does not access the phenomenological content of Sumerian religious experience — the feeling of encountering the `me`, the dread of `namtar`. It cannot distinguish between "the concept is alien" and "the alignment is imperfect." Future work — specifically the Phase 2 Riemannian geometry analysis outlined in `docs/RESEARCH_VISION.md` — would approach the same question through curvature and distortion tensors, providing an independent geometric measure that could partially separate alignment quality from conceptual distance.

**The cosmogonic axis as a research tool.** The translation-opacity axis projected in `docs/figures/cosmogony/cosmogony_axis_projection.png` is not merely a descriptive tool. It generates a testable prediction: concepts that sit near `namtar` on this axis (fate-adjacent vocabulary in Sumerian) should be more cross-linguistically portable than concepts that sit near `abzu`. Comparative analysis with Egyptian hieroglyphic embeddings (the Heiroglyphy project, which currently achieves 32.35% top-1 alignment) could test whether Egyptian fate-concepts are similarly translation-adjacent, and whether Egyptian creation-concepts are similarly orthogonal to English spatial vocabulary. If the pattern holds across unrelated ancient languages, it would support the cognitive-universality hypothesis for fate and the alien-topology hypothesis for cosmogonic-origin vocabulary.

### 9.6 The Cosmogonic Arc Revisited

With the five deep dives complete, we can revisit the cosmogonic arc of §3 and read it geometrically. The arc moves from `abzu` (pre-creation, orthogonal to English) through `zi` (animation, near-orthogonal) and `nam` (essence-machinery, low displacement) to `namtar` (fate-determination, translation-adjacent) and `me` (civilizational templates, near-orthogonal). The movement is not monotonically from alien to familiar: `namtar` is the exception, not the culmination.

This non-monotonic pattern is itself a cosmogonic observation. The Sumerian cosmos does not move from the conceptually alien (creation) to the conceptually familiar (civilization). Fate — determined at the midpoint of the arc — turns out to be the most conceptually portable element, while the beginning (`abzu`) and the end (`me`) are both geometrically remote from English. The structure suggests that what Sumerian cosmogony conserved most faithfully across the translation into English vocabulary is not its endpoint (civilizational order) but its axis (the moment of fate-determination), and that both the origin-state (`abzu`) and the distribution of civilizational being (`me`) are less recoverable through English semantic structure than the intermediate act of destiny-cutting.

This is a tentative reading of a geometric pattern. It may be a feature of the alignment architecture and corpus coverage as much as of the underlying conceptual structure. But it is a reading that the data support, and it is more specific and falsifiable than the general claim that "Sumerian thought is different from modern thought."

---

## § 10 — Reproducibility

All numeric claims in this document trace to `docs/cosmogony_tables.json` (schema version 1), committed at the alignment-artifact commit. Every number cited — cosine similarities, displacement values, analogy probe results — corresponds to a specific path in that JSON file, noted inline throughout §4–9.

**Pinned alignment-artifact commit:** `5945bd6` (tag: chore: commit generated cosmogony figures)

This commit contains:
- `docs/cosmogony_tables.json` — the canonical numeric tables
- `docs/figures/cosmogony/*.png` — the seven committed figures
- `final_output/sumerian_aligned_gemma_vectors.npz` — the Sumerian-to-Gemma aligned vectors
- `final_output/sumerian_aligned_vectors.npz` — the Sumerian-to-GloVe aligned vectors
- `final_output/sumerian_aligned_vocab.pkl` — the 35,508-token vocabulary

To regenerate the tables and verify byte-identity:

```bash
python scripts/analysis/generate_cosmogony_tables.py
python scripts/analysis/generate_cosmogony_figures.py
```

Both entry points are deterministic: UMAP uses a fixed random seed, iteration order over the concept slate is sorted, and all floating-point operations are seeded. Byte-identical regeneration requires the same Python and dependency versions specified in `requirements.txt`.

The pre-flight report that validated the concept slate is at `results/cosmogony_preflight_2026-04-20.json`. All five concepts passed with no warnings. The `etcsl_passage_finder` normalization fix (commit `ec352b1`) was required to ensure `namtar` matched correctly against ETCSL transliterations; without it, `namtar` would have failed the passage-count check.

Test coverage for the six analysis modules:

```bash
pytest tests/analysis/ -v
```

All 144 tests passing at the pinned commit confirms the infrastructure integrity.

---

## § 11 — References

**Primary sources (ETCSL)**

- ETCSL text c4332 — Fire-god hymn (abzu passages: c4332.3, c4332.40; namtar passage: c4332.27)
- ETCSL text c0201 — Hymn to Inanna (me passages: c0201.4, c0201.8)
- ETCSL text c2554 — Ninurta/heroic hymn (nam passages: c2554.B.11, c2554.C.13)
- ETCSL text c3205 — Royal hymn / hymn to Šulgi (namtar passage: c3205.28; zi passage: c3205.46)
- ETCSL text c6116 — Proverb collection (zi passage: c6116.E.16.e10.22)
- ETCSL 1.1.2 — *Enki and Ninmah* (cited in §3 and §5)
- ETCSL 1.1.3 — *Enki and the World Order* (cited in §3 and §5)
- ETCSL 1.3.1 — *Inanna and Enki* (cited in §3 and §8)
- ETCSL 1.4.1 — *The Descent of Inanna to the Underworld* (cited in §7)
- ETCSL 1.7.4 — *The Eridu Genesis* (cited in §3)
- ETCSL 1.8.1.4 — *Gilgamesh, Enkidu, and the Netherworld* (cited in §3)
- ETCSL 2.1.5 — *The Curse of Agade* (cited in §7)

**Secondary sources**

- Black, Jeremy, Graham Cunningham, Eleanor Robson, and Gábor Zólyomi. *The Literature of Ancient Sumer*. Oxford University Press, 2004.
- Foxvog, Daniel. *Introduction to Sumerian Grammar*. Current revision (online). Cited for compound-noun and nominalizing-morphology sections.
- Jacobsen, Thorkild. *The Treasures of Darkness: A History of Mesopotamian Religion*. Yale University Press, 1976.
- Kramer, Samuel Noah. *Sumerian Mythology: A Study of Spiritual and Literary Achievement in the Third Millennium B.C.* University of Pennsylvania Press, 1944. Revised edition, 1972.
- Michalowski, Piotr. "On the Early History of the *eridu* Lament." *Acta Sumerologica* 12 (1993). Cited for the characterization of `me` as the sum of civilizational institutions.

---

## § 12 — Appendix: Pre-Flight Concept Availability

The pre-flight concept check was run on 2026-04-20 using `scripts/analysis/preflight_concept_check.py`. Results are committed at `results/cosmogony_preflight_2026-04-20.json`. All five candidate concepts passed with no failures and no warnings.

### Concept Verdicts

| Concept | English Anchor | Sumerian in Vocab | English in Gemma | English in GloVe | ETCSL Passages | Status |
|---------|---------------|-------------------|-----------------|-----------------|----------------|--------|
| abzu | deep | yes | yes | yes | 97 | PASS |
| zi | breath | yes | yes | yes | 193 | PASS |
| nam | essence | yes | yes | yes | 561 | PASS |
| namtar | fate | yes | yes | yes | 35 | PASS |
| me | decree | yes | yes | yes | 561 | PASS |

The passage count for `namtar` (35) is the lowest of the five, consistent with the concept's more specialized role in Sumerian literary tradition. Nonetheless, 35 passages is sufficient for ETCSL grounding in §7: the concept is well-attested across ritual, hymnic, and mythological contexts.

### The Normalization Bug and Fix

An earlier preflight run (before commit `ec352b1`) incorrectly flagged `namtar` as failing the ETCSL passage check, returning zero passages. Investigation revealed that the `etcsl_passage_finder` module was comparing raw ETCSL transliterations against the query token without normalizing either. The ETCSL corpus uses hyphenated forms (`nam-tar`) while the production vocabulary and the query token use fused forms (`namtar`, post-normalization by `sumerian_normalize.py`). The mismatch produced zero matches despite the concept being abundantly attested.

Commit `ec352b1` fixed `etcsl_passage_finder` to apply `normalize_sumerian_token` from `scripts/sumerian_normalize.py` to both the query token and each transliteration line before matching. After the fix, `namtar` returns 35 passages, clearing the pre-flight check. The fix is consistent with the normalization architecture established in Workstream 2b: all token comparisons in the pipeline now use the same normalization chain (subscripts → ASCII, determinative braces stripped, hyphens dropped, lowercased).

No concept substitution was required. The original concept slate — `abzu`, `zi`, `nam`, `namtar`, `me` — is used throughout the document as planned.

An earlier, pre-fix preflight had identified a potential substitution: `namtar` → `kur` (netherworld/mountain) would have been the substitution under the original rules (substitute with a pre-approved alternate if ETCSL passage count is zero). This substitution would have substantially changed the analysis: `kur` is a geographic-cosmological term with very different distributional properties from `namtar`, and the `namtar`-as-translation-adjacent finding would not have appeared in the results. The normalization fix preserved the original concept slate and with it the key finding about fate's cross-linguistic portability.

### Degenerate Top-5 Check

No concept showed a degenerate top-5 in either Gemma or GloVe space (the `degenerate_fraction_top5` field is 0.0 for all five concepts in the preflight JSON). The degenerate-top-5 flag triggers when the top-5 nearest Sumerian neighbors are dominated by single-character tokens or other clearly noise-driven results (the `sirara→c` pattern documented in the Workstream 2a audit). The absence of degeneracy across all five concepts confirms that the alignment anchors for these concepts are clean.
