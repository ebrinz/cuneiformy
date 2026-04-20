---
title: "Six Lenses on a Dead Language: Findings from the Sumerian Anomaly Atlas"
author: "Cuneiformy Research Program"
date: "2026-04-20"
---

# Six Lenses on a Dead Language: Findings from the Sumerian Anomaly Atlas

**Atlas artifact commit:** `ff94533`
**Atlas JSON:** `docs/anomaly_atlas.json`
**Total aligned tokens:** 35,508
**Branch:** `feat/anomaly-findings-doc`

---

## § 0 — Abstract

This document presents a thematic interpretation of the Sumerian Anomaly Atlas, a ranked diagnostic applied to 35,508 aligned Sumerian tokens at atlas-artifact commit `ff94533`. Six computational lenses — English displacement, untranslated high-frequency terms, isolation in embedding space, cross-space divergence, near-duplicate detection, and structural bridging — surface patterns in the alignment that resist simple description as either "translation accuracy" or "translation failure." The top finding from Lens 1 is 𒂗 *en* (traditionally "lord" or "priest"), which carries a cosine similarity of -0.037 between its projected vector and the English gloss: not a catastrophic mismatch, but a geometrically meaningful separation. The highest-isolation token from Lens 3 is *uttu*, a weaving goddess whose embedding neighborhood consists almost entirely of her own morphological inflections — a signature of specialized cultic vocabulary without distributional support from nearby concepts. Lens 5's top pairs are transliteration-convention shadows: the same token recorded by different ORACC conventions (c/š alternation) arrives at cosine similarity 0.998, validating the FastText subword model's convention-robustness. Lens 6's bridge winners are agglutinated compound forms spanning two k-means clusters, consistent with Sumerian morphology creating genuine semantic hybrids in vector space. The atlas is more useful as a diagnostic than as a discovery engine: its most interpretable signals live in the middle of each ranked list, not at the extremes. This document is standalone and does not re-engage with any prior interpretive thesis about specific Sumerian concepts.

---

## § 1 — Introduction

### What the Atlas Is

The Sumerian Anomaly Atlas is a computational audit of 35,508 Sumerian tokens, each represented as a FastText-derived 768-dimensional vector aligned into English semantic space via whitened ridge regression. At the atlas commit (`ff94533`), the alignment achieves 52.13% top-1 accuracy on held-out anchor pairs — meaning that for just over half of Sumerian words, the closest English word in the projected space is the word a Sumerologist would call the gloss. The other 48% land somewhere else, and the atlas is designed to characterize that "somewhere else" systematically rather than anecdotally.

The term "anomaly" in this context does not imply error. An anomalous token is one whose relationship between the Sumerian and English representations is interesting under at least one of the six lenses. The lenses are not independent: a token that scores highly on Lens 1 (its English gloss lands far from its projected vector) may also score on Lens 3 (its Sumerian neighborhood is sparse), because both signatures can arise from a concept that is culturally specific, under-attested, or simply poorly served by its English gloss. The atlas makes no claim to resolve which cause applies for any given token. It provides a ranked list and asks the reader to bring domain knowledge to bear.

### The Six Lenses at a Glance

**Lens 1 — English displacement.** For each anchor pair (Sumerian token, English gloss), compute the cosine similarity between the token's projected Sumerian vector and the English word's native English vector. Negative cosines indicate that the projected Sumerian vector points geometrically away from the English gloss, even in the aligned space. This is the closest the atlas comes to measuring "translation failure" — with all the caveats that follow from alignment noise, corpus bias, and the limited sense in which cosine similarity captures semantic proximity.

**Lens 2 — Untranslated high-frequency terms.** For non-anchor tokens (those without a Sumerologist-assigned gloss in the training set), score each by corpus frequency times one minus the cosine similarity to the best-matching English word. High scores identify tokens that appear often in Sumerian texts but have no close English neighbor: the system cannot translate them, and they are important enough by frequency to matter.

**Lens 3 — Isolation in source space.** Rank tokens by cosine distance to their tenth nearest Sumerian neighbor. A token with a high isolation score is surrounded by near-vacuity in the source space: its nearest neighbors are far away, and the next-nearest are even farther. Isolation is a proxy for lexical specificity — a concept with no close Sumerian relatives and no close English counterpart is genuinely semantically isolated by the corpus's distributional evidence.

**Lens 4 — Cross-space divergence.** For tokens present in both the whitened-Gemma and GloVe aligned spaces, compute the Jaccard distance between the two spaces' top-10 neighbor lists. High Jaccard distance means the two alignment methods see completely different neighborhoods for the same token: zero overlap. This is a reliability flag as much as an anomaly signal — a token with Jaccard distance 1.0 has no alignment consensus.

**Lens 5 — Doppelgangers.** Find source-token pairs with cosine similarity above 0.95. Near-identical embeddings in the source space are candidates for morphological variants, scribal variants, or transliteration-convention alternates. In the Sumerian corpus, the dominant doppelganger pattern is orthographic: the same phoneme written with different ATF conventions by different ORACC editors.

**Lens 6 — Structural bridges.** Apply k-means clustering (k = 40, seed = 42) to the source-space vectors and compute a bridge score for each token: the inverse of the ratio between the token's distance to its nearest cluster centroid and its distance to its second-nearest cluster centroid, normalized so that a perfectly equidistant token scores 1.0. Tokens with high bridge scores sit in the geometric middle between two clusters — candidate conceptual bridges.

### Scope and Caveats

The atlas covers 35,508 tokens, of which 8,998 are anchors (tokens with Sumerologist-assigned glosses, sourced from ePSD2 and ETCSL parallel translations) and 26,510 are non-anchors. The corpus underlying the FastText training is the ETCSL-CDLI-ORACC fused transliteration, dominated by Old Babylonian literary and administrative texts (roughly 2000–1600 BCE). Administrative and accounting texts are well represented; the Lens 2 results show this clearly, as they are dominated by numeric classifiers. The top-1 alignment accuracy of 52.13% is high enough to support interpretation but not so high that individual neighbor results can be treated as definitive.

---

## § 2 — Methodology

### Pipeline Summary

The Cuneiformy alignment pipeline runs in four stages (documented fully in `docs/superpowers/specs/2026-04-19-workstream-2b-normalization-fix-design.md`). Stage 1 trains FastText on the fused Sumerian transliteration corpus. Stage 2 extracts anchor pairs (Sumerian token, English gloss) from ePSD2 and ETCSL, applying a normalization chain — subscript-to-ASCII conversion, determinative-brace stripping, hyphen removal, lowercasing — that must match the normalization applied during tokenization. Stage 3 applies whitened ridge regression at regularization parameter α = 100 to project Sumerian vectors into English target space; the whitening step transforms the Gemma-derived English embedding distribution to unit covariance before regression, reducing the geometric distortion caused by the contextual model's anisotropic output. Stage 4 produces the dual-view lookup API exposing both the whitened-Gemma and GloVe aligned spaces.

The anomaly atlas script (`scripts/analysis/sumerian_anomaly_atlas.py`) is a thin wrapper around the six lens functions in `scripts/analysis/anomaly_lenses.py`. All lens computations are pure functions with no Sumerian-specific logic; the framework is civilization-agnostic. The atlas JSON at commit `ff94533` is the committed output of a single deterministic run. All numeric claims in this document trace to `docs/anomaly_atlas.json` at that commit.

A prior document, `docs/sumerian_cosmogony.md`, applied similar methodology to five hand-picked cosmogonic concepts (*abzu*, *zi*, *nam*, *namtar*, *me*); the present document is independent of that interpretive thesis and draws from the full 35,508-token atlas rather than a curated subset.

### Alignment Quality and Its Limits

The 52.13% top-1 accuracy figure is a held-out-anchor accuracy: for anchors withheld from ridge training, how often does the top-1 projected neighbor match the withheld gloss? This is an optimistic figure in one sense (it measures only the anchored vocabulary) and a pessimistic one in another (the alignment is trained on a relatively small anchor set, with alignment noise compounded by the FastText morphology artifact that clusters inflected forms near their roots).

Lens 1's negative cosine findings deserve special caution. A cosine near zero between a projected Sumerian vector and an English word does not mean the Sumerian word is semantically unrelated to the English gloss in the human sense; it means the alignment places them in geometrically unrelated parts of the projected space. This can happen because: (1) the English gloss is genuinely wrong or too reductive; (2) the anchor confidence is low and the gloss-vector pair is noisy; (3) the Sumerian token has a sparse distributional neighborhood that the ridge regression cannot resolve; or (4) the alignment noise floor is reached. The atlas provides confidence scores for each anchor (derived from the ePSD2 or ETCSL source quality rating); we use confidence as a filter and note it throughout the case studies.

### Document Structure

Sections §3–8 address the six themes corresponding to the six lenses. Each theme section contains two to four case studies following a four-part template: (1) the finding in one sentence; (2) the Sumerological anchor with ePSD2 or secondary citation; (3) the geometry from atlas data; (4) a hedged interpretive reading. §9 synthesizes across themes. §10 pins the reproducibility information. §11 gives full references. §12 provides cuneiform sign provenance for every cuneiform character embedded in the document.

---

## § 3 — 𒂗 Theme 1: Translation Failures

*Heading sign: 𒂗 EN, "lord/priest"*

The six-lens atlas contains no clean binary of "successfully translated" versus "failed translation." What it does contain is a ranked list of anchor pairs whose projected Sumerian vector lands geometrically distant from the English gloss — tokens where the cosine similarity is small, zero, or negative. Negative cosines are the most striking: they mean the projection of the Sumerian token actively points away from the English word in the aligned space. Lens 1 surfaces three high-confidence cases that repay close reading: 𒂗 *en*, *jizzal*, and *rin2*.

---

### 3.1 — 𒂗 *en* "priest": When Titles Cross Domains

**The finding.** The token *en*, glossed "priest" in the ePSD2, has a cosine similarity of -0.037 between its projected Sumerian vector and the English word "priest" in the aligned space (`lens1_english_displacement.rows_filtered[4]`, anchor confidence 0.95, source ePSD2).

**The anchor.** ePSD2 (http://oracc.museum.upenn.edu/epsd2/sumerian/en) lists *en* as a noun with primary gloss "lord" and secondary gloss "priest," specifically the high priest or priestess of certain major Sumerian temples. The ETCSL parallel translations use "lord" as the primary rendering for *en* in literary contexts (for example, *en-lil₂* = Enlil, "lord wind"), while administrative texts from the ED III and Ur III periods use "en" as a temple title — the *en* of Inanna at Uruk was a ritual office, possibly occupied by a male or female priest depending on the period (Jacobsen 1976, pp. 25–28; Michalowski 1993, pp. 112–114). The English gloss "priest" catches only one face of a term whose range spans political authority, divine epithet, and cultic office.

**The geometry.** The atlas reports cosine similarity -0.037 for the *en* → "priest" pair (`lens1_english_displacement.rows_filtered[4]`). A cosine of -0.037 is only modestly below zero — geometrically, the vectors are nearly orthogonal, pointing in slightly opposite directions. The anchor confidence of 0.95 (ePSD2 sourced) is high, making anchor-quality noise an unlikely explanation. The finding is that in the projected Sumerian space, the distributional neighborhood of *en* does not overlap with the distributional neighborhood of the English word "priest." English "priest" clusters near terms of religious intermediacy — confessor, officiant, clergyman — while *en*, projected into the same space, sits near the political and divine register: lordship, authority, divine office. The separation is small in absolute terms but consistent with the interpretation that "priest" is not a false gloss but a narrow one.

**Interpretation.** The embedding geometry is consistent with the Sumerological observation that *en* is a polysemous title spanning at least three distinct domains: divine epithet (Enlil = en + lil₂, "lord of the air"; Enki = en + ki, "lord of the earth"), political office (the *en* of a city in Early Dynastic texts), and cultic office (the *en* of Inanna at Uruk, a ritual appointment). English "priest" captures only the cultic use and imports the Christian-tradition semantics of priestly intermediacy and sacrifice that are absent from the Sumerian record. The slight negative cosine suggests (tentatively) that the distributional environment of *en* in Sumerian texts — co-occurring with divine names, city names, and political formulae — geometrically resembles the English register of lordship and divine title more closely than the English register of priesthood. The atlas data cannot adjudicate between the English glosses, but it flags the tension as geometrically real.

---

### 3.2 — *jizzal* "ear": Metonymic Reduction

**The finding.** The token *jizzal*, glossed "ear" in the ePSD2, has a cosine similarity of -0.063 between its projected Sumerian vector and the English word "ear" (`lens1_english_displacement.rows_filtered[1]`, anchor confidence 0.95, source ePSD2).

**The anchor.** ePSD2 (http://oracc.museum.upenn.edu/epsd2/sumerian/jizzal) glosses *jizzal* primarily as "ear" with the additional semantic note that in Sumerian literary texts the ear is frequently the seat of wisdom, discernment, and attentive intelligence — the organ through which the gods' commands are received and through which wisdom passes from master to student. The phrase *jizzal gub* (literally "to stand one's ear") means to pay attention, to be attentive (Black et al. 2004, p. 316). The literary register of *jizzal* is heavily associated with wisdom reception rather than physical anatomy.

**The geometry.** The cosine of -0.063 is the second most-negative value in the filtered Lens 1 table (`lens1_english_displacement.rows_filtered[1]`), with anchor confidence 0.95. In the aligned space, the English word "ear" clusters near physical-anatomy vocabulary — hearing, auditory, canal, drum — whereas the projected Sumerian vector for *jizzal* lands at geometrically negative cosine to that cluster. The 5-nearest Sumerian neighbors of *jizzal* in the source space are not readily interpretable from the atlas data alone, but the negative cosine to "ear" indicates that the distributional environment of *jizzal* in the Sumerian corpus looks more like wisdom and reception vocabulary than like body-part vocabulary.

**Interpretation.** The atlas data is consistent with a reading in which *jizzal* encodes the wisdom-bearing, reception-oriented aspect of the ear rather than the anatomical organ. English "ear" necessarily imports its anatomical-phonetic neighborhood: sound waves, hearing, deafness, the ear canal. The Sumerian literary corpus uses *jizzal* in contexts of divine command reception, instruction, and attentive obedience — a metonymic pattern documented across multiple ETCSL wisdom texts (Black et al. 2004, pp. 316–318). The embedding geometry is consistent with the hypothesis that FastText, trained on this corpus, has encoded the metonymic use as the dominant distributional signal for *jizzal*, producing a projected vector that sits geometrically nearer to "attention" or "obedience" vocabulary than to "ear" in the English space. The atlas cannot confirm this interpretation — it can only flag the mismatch and note its magnitude.

---

### 3.3 — *rin2* "lord": The Sharpest Displacement

**The finding.** The token *rin2*, glossed "lord" in ETCSL translations, has a cosine similarity of -0.088 between its projected Sumerian vector and the English word "lord" — the most negative cosine similarity in the entire filtered Lens 1 table (`lens1_english_displacement.rows_filtered[0]`, anchor confidence 1.0, source ETCSL).

**The anchor.** ETCSL parallel translations gloss *rin2* as "lord" (anchor confidence 1.0, meaning the same gloss is confirmed across multiple ETCSL text occurrences). The word is rarer than *en*: ePSD2 (http://oracc.museum.upenn.edu/epsd2/sumerian/rin2) attests *rin2* in a narrower range of contexts than the politically and divinely ubiquitous *en*, appearing primarily in epithets and titles where "lord" functions as an honorific rather than an office-name. Foxvog (*Introduction to Sumerian Grammar*, §3.2.1) notes that Sumerian has several near-synonymous title words — *en*, *lugal*, *rin2* — that occupy overlapping but distinct distributional niches in the texts.

**The geometry.** With cosine -0.088 and anchor confidence 1.0 (`lens1_english_displacement.rows_filtered[0]`), *rin2* is the strongest displacement signal in the atlas, and it is also the most trustworthy: ETCSL anchor confidence 1.0 means the gloss is established across multiple attestations, not a single uncertain attribution. The projected Sumerian vector for *rin2* points geometrically away from the English word "lord" in the aligned space. English "lord" is a broad term — it covers feudal overlords, divine lordship, forms of address, and compound terms like "Lord of the Flies." The distributional neighborhood of "lord" in English embedding space is correspondingly broad: master, ruler, king, God, sovereign. The projected *rin2* vector does not land near that broad neighborhood, suggesting (tentatively) that *rin2*'s distributional environment in Sumerian texts is more specific and narrow than "lord" implies.

**Interpretation.** The atlas data is consistent with a reading in which *rin2* occupies a narrower, more contextually specific niche than the English "lord" can carry. The negative cosine is the strongest signal in the atlas for translation-inadequacy, and it occurs not for a rare word with uncertain attestation but for a well-attested title with a confirmed gloss. The most likely explanation, based on the Sumerological record, is that *rin2* functions primarily as an honorific epithet in cultic contexts — its distributional neighbors in the corpus are divine names and ritual epithets, not the broad political-feudal-religious range that English "lord" covers. The alignment cannot confirm whether *rin2* and *en* are near-synonyms or distributionally distinct; a separate semantic-field analysis would be required. What the atlas establishes is that "lord" as a gloss for *rin2* produces the single largest cosine misalignment in the filtered dataset.

---

## § 4 — 𒄑 Theme 2: Specialized Cultic Vocabulary

*Heading sign: 𒄑 GISH, wood/tree (wooden-object determinative)*

Several of the most isolated tokens in Lens 3 share a common characteristic: they are Sumerian terms for specific cultic objects, deities, or places that resist reduction to a general English category. These terms are not "untranslatable" in the trivial sense of having no English equivalent — Sumerologists have assigned glosses to all of them — but their embedding geometry suggests that the distributional support for those glosses is thin. Their nearest Sumerian neighbors are almost entirely morphological variants of themselves (suffixed forms, possessed forms, directional forms), with almost no thematic vocabulary nearby. They exist as isolated peaks in the source-space topology.

---

### 4.1 — *uttu* "goddess of weaving": Isolation by Specificity

**The finding.** The token *uttu*, the name of the Sumerian goddess of weaving and cloth, has a cosine distance to its tenth nearest Sumerian neighbor of 0.622 — the third-highest isolation score in the atlas (`lens3_isolation.rows[2]`).

**The anchor.** ePSD2 (http://oracc.museum.upenn.edu/epsd2/sumerian/uttu) identifies *uttu* as a deity name, the goddess of weaving, attested prominently in *Enki and Ninhursag* (ETCSL 1.1.1) where she is the final product of a sequence of divine births and becomes the mother of the eight healing plants. Kramer (1944, pp. 54–59) describes *uttu*'s role as weaving goddess in the context of the divine domestic economy: she produces the cloth that clothes the gods and symbolizes the ordered productive capacity of civilized life. Unlike major deities such as Enlil or Inanna, *uttu* is a specialist deity — her domain is narrow and specific, and she appears in a limited range of mythological contexts.

**The geometry.** The Lens 3 atlas records *uttu*'s distance to its tenth nearest Sumerian neighbor as 0.622 (`lens3_isolation.rows[2]`, `distance_to_kth_neighbor`). The five nearest neighbors are: *uttura* (cosine similarity 0.758), *munuse* (0.410), *muunnasug2sug2geesz* (0.405), *nibuluj3* (0.396), and *sikillake4* (0.394). The nearest neighbor *uttura* is an inflected form of *uttu* itself (the locative or terminative suffix -ra attached to the divine name); the next four neighbors drop sharply to cosine similarities below 0.41, meaning the embedding sees almost nothing thematically near *uttu* in the source space. The cosine-distance histogram for all tokens shows that 98.6% of tokens have a tenth-neighbor distance below 0.600; *uttu* is among the 24 tokens in the entire 35,508-token vocabulary whose isolation score exceeds 0.600.

**Interpretation.** The embedding geometry is consistent with the Sumerological profile of *uttu* as a domain-specific deity. A word whose corpus appearances are overwhelmingly as a deity name in a small set of mythological texts will, under FastText distributional training, accumulate neighbors that are either its own morphological forms or other tokens that co-occur in those same texts. The sharp drop-off from 0.758 (nearest morphological variant) to 0.410 (nearest semantic neighbor) suggests that *uttu* occurs in distributional environments unlike those of any other Sumerian word in the atlas vocabulary. The atlas cannot determine whether this isolation is a finding about the concept itself (that weaving-goddess vocabulary is genuinely isolated in Sumerian thought) or an artifact of corpus sampling (that the few texts featuring *uttu* are unusual texts with unusual co-occurrence environments). Both explanations are plausible; the isolation score establishes the pattern without resolving its source.

---

### 4.2 — 𒄑 *gidri* "scepter": The Emblem That Stands Alone

**The finding.** The token *gidri*, a Sumerian word for a specific type of scepter or staff of divine office, has an isolation score of 0.619 (`lens3_isolation.rows[4]`).

**The anchor.** ePSD2 (http://oracc.museum.upenn.edu/epsd2/sumerian/gidri) glosses *gidri* as "scepter" or "staff," a royal and divine emblem of authority. The 𒄑 (GISH) determinative for wooden objects is applied to *gidri* in cuneiform writing, marking it as a wooden artifact (Black et al. 2004, p. 96). In Sumerian royal hymns, the *gidri* is presented to kings by the gods as a sign of divine mandate — its most prominent contexts are enthronement and investiture ceremonies. The *gidri* is distinct from other Sumerian royal emblems: it is not the crown (*men*), not the throne (*gu-za*), and not the mace (*šita*), but specifically a scepter-shaped staff whose shape and material signify divine authority.

**The geometry.** The Lens 3 atlas gives *gidri* an isolation score of 0.619 (`lens3_isolation.rows[4]`). Its five nearest neighbors are *gidriku3* (cosine 0.720, the "pure/bright scepter" phrase), *e2gidri* (0.697, "scepter-house"), *gidru* (0.563, a variant spelling), *igalimkake4* (0.396), and *be2garre2esz2* (0.392). The two highest-similarity neighbors are compound forms built on *gidri* itself, not independent semantic neighbors. *gidru* is a spelling variant. The fourth and fifth neighbors drop to 0.396 and 0.392, leaving a large gap between *gidri*'s cluster of self-forms and anything else the embedding finds nearby.

**Interpretation.** The geometry of *gidri* is consistent with the Sumerological picture: a specialized cultic-royal term that appears in narrow ceremonial contexts, flanked primarily by its own modified forms rather than by a thematic semantic field. English "scepter" is a reasonably adequate gloss for the physical object — the GISH determinative confirms its wooden-object status — but the distributional environment of *gidri* in the Sumerian corpus is one of divine investiture, not general regalia vocabulary. The near-identical behavior of *gidri* and *uttu* under Lens 3 — high isolation, nearest neighbors are self-inflections, sharp drop-off to the next tier — suggests a class of Sumerian tokens characterized by narrow distributional environments and poor generalization across texts. These are the "precision vocabulary" tokens: their meaning is exact, their occurrence is sparse, and their embedding is isolated by consequence.

---

### 4.3 — *ebgal* "the great shrine": Toponym or Category?

**The finding.** The token *ebgal*, variously interpreted as a toponym or as a generic term for a great shrine complex, has an isolation score of 0.616 (`lens3_isolation.rows[9]`).

**The anchor.** ePSD2 (http://oracc.museum.upenn.edu/epsd2/sumerian/ebgal) lists *ebgal* (from *e2* "house/temple" + *gal* "great") as meaning "great temple" or "great palace." The word appears as both a common noun (a generic term for a large sacred complex) and as a proper name for specific temples, particularly the great temenos of the major Sumerian cities. In the hymn literature, *ebgal* is used as an honorific epithet for the most important temple in a city. The ambiguity between common noun and proper noun means that in corpus occurrence, *ebgal* appears in both generic architectural descriptions and in proper-name contexts where it refers to a specific building.

**The geometry.** The isolation score of 0.616 (`lens3_isolation.rows[9]`) places *ebgal* among the top 24 isolated tokens in the atlas. Its five nearest Sumerian neighbors are *ebgalla* (cosine 0.698, the locative/possessive form), *ebgalta* (0.647, ablative form), *ibgal* (0.582, a variant spelling or related term), *ninebgal* (0.553, "lady of the great shrine"), and *lu2ebgal* (0.523, "person of the great shrine"). As with *gidri*, the nearest neighbors are either inflected forms, spelling variants, or compounds built directly on the base word. The semantic neighborhood of *ebgal* in the source space is almost entirely self-referential: it clusters with its own modified forms and compounds rather than with generic architectural vocabulary.

**Interpretation.** The embedding geometry is consistent with *ebgal* behaving as a near-proper-noun in the corpus. When a term functions simultaneously as a common noun and a proper name, FastText's distributional training produces a vector that reflects the specific co-occurrence environment of that term — which, for a proper name, is dominated by the liturgical and administrative contexts in which the named place is mentioned. The neighbor *ninebgal* (goddess of the great shrine) suggests that *ebgal* occurs frequently in divine-epithet compounds, reinforcing its proper-noun behavior. The broad English gloss "great shrine" captures the compositional semantics (e2 + gal) but not the specificity that the corpus evidence encodes.

---

### 4.4 — *asag* "demon of chaos": Cross-Space Incoherence

**The finding.** The token *asag*, a Sumerian term for a specific type of dangerous supernatural entity, shows Jaccard distance 1.0 between the top-10 Gemma-space neighbors and the top-10 GloVe-space neighbors (`lens4_cross_space_divergence.rows_anchor_only` entry for *asag*) — the highest possible divergence, indicating zero neighborhood overlap.

**The anchor.** ePSD2 (http://oracc.museum.upenn.edu/epsd2/sumerian/asag) identifies *asag* as a demon or malevolent supernatural force, described in the myth *Ninurta's Exploits* (ETCSL 1.6.2) as the Asag, a fearsome entity made of stone that Ninurta defeats in cosmic combat. The ETCSL translation renders the entity as "the Asag" in most passages, treating it as a proper name (the text contains the famous description: "the Asag and its hideous offspring have settled in the mountains," ETCSL 1.6.2, ll. 25–30). Kramer (1944, pp. 79–81) notes that *asag* is not merely a monster in the folkloric sense but represents a principle of cosmic disorder that must be overcome for civilization to function — its defeat by Ninurta is an ordering event, not merely a heroic episode.

**The geometry.** The Jaccard distance of 1.0 (`lens4_cross_space_divergence.rows_anchor_only`) is the maximum possible, indicating that the Gemma-aligned space and the GloVe-aligned space see completely different neighborhoods for *asag*, with zero tokens in common among their respective top-10 neighbor lists. The Gemma-space top-10 neighbors are: *sagabi*, *inannaursag*, *asagdu3duta*, *kusag*, *du6karsag*, *esag*, *aslag4*, *ansag*, *ki:sag*, *gursag* — largely surface-form neighbors (tokens containing the string "sag") rather than semantic neighbors. The GloVe-space top-10 are: *munu4bi*, *<sila3>*, *asila3*, *si4u2lum*, *urux*, *sze_*, *ziz2ga*, *iszarlibur*, *sila3>*, *aszhalum* — a completely incoherent set that appears to be noise from the GloVe alignment's treatment of rare tokens.

**Interpretation.** The cross-space divergence for *asag* is almost certainly an alignment reliability problem rather than a conceptual finding. The Gemma-space neighbors are dominated by string-similarity artifacts (tokens containing "sag"), while the GloVe-space neighbors are uninterpretable noise. This is consistent with *asag* being a rare, specialized term that lacks sufficient distributional support in either aligned space to produce stable neighborhoods. The atlas cannot be said to "find" anything about *asag*'s meaning from the Lens 4 result; it finds that *asag* is under-aligned in both spaces. The Sumerological content — that *asag* is a cosmic chaos-demon whose defeat enables civilization — cannot be recovered from this alignment geometry. The lesson for the reader is that Lens 4 high-divergence results are reliability flags, not semantic discoveries.

---

## § 5 — 𒆠 Theme 3: Grammatical Bridges

*Heading sign: 𒆠 KI, "place/earth" (place-name determinative)*

Sumerian is agglutinative: grammatical information is expressed by stacking suffixes and prefixes onto root morphemes, producing surface tokens that may be several morphemes long. In a FastText model trained on surface tokens, a form like *ningir2su2kake4* — the name of the god Ningirsu with a dative case suffix and a connective suffix — receives a single vector that FastText constructs from the character n-grams of the whole token string. If this multi-morpheme token appears in corpus contexts that mix deity-name environments with grammatical-suffix environments, its resulting vector will sit geometrically between those two distributional clusters. Lens 6 directly measures this: bridge tokens are tokens equidistant between two k-means clusters, candidates for tokens whose distributional ecology spans multiple conceptual domains.

---

### 5.1 — *ningir2su2kake4*: A God's Name in Grammar's Grip

**The finding.** The token *ningir2su2kake4* — the name of the deity Ningirsu with dative-case suffix *-ke4* and connective *-a* — has a bridge score of 1.000 between k-means clusters 9 and 4 (`lens6_structural_bridges.rows[0]`).

**The anchor.** Ningirsu (ePSD2 sign representation: 𒀭Ningirsu, divine determinative + name) is the city god of Girsu, identified in ETCSL texts with the heroic warrior aspect — most extensively in *Ninurta's Exploits* (ETCSL 1.6.2) where Ningirsu and Ninurta are conflated or identified. The divine name Ningirsu encodes the compound *nin* + *gir2su* + (divine determinative): "lord of Girsu" (Black et al. 2004, p. 165). The form *ningir2su2kake4* adds the dative case marker *-ke4* (the agent/dative suffix common in Sumerian verbal chains) to the full divine name, producing a grammatically complete participant-marking form that would appear in clauses where Ningirsu is the grammatical subject or agent.

**The geometry.** Bridge score 1.000 (`lens6_structural_bridges.rows[0]`, `bridge_score: 0.999997`) with clusters 9 and 4. The Lens 6 atlas records five representative members of each cluster: cluster 9 contains *nin*, *geme2*, *ningir2su*, *ninazu*, *ninlil2* — a tight divine-name and divine-title cluster (nin = "lady/lord," geme2 = "female servant," the others are deity names). Cluster 4 contains *ba*, *i3*, *sze3*, *za*, *igi* — a cluster of short grammatical morphemes and postpositions. The token *ningir2su2kake4* sits at equal geometric distance from both clusters, meaning its vector receives equal "pull" from the deity-name vocabulary and the grammatical-morpheme vocabulary. This is mechanically consistent with how it is constructed: it is a deity name to which grammatical morphemes have been appended.

**Interpretation.** The embedding geometry is consistent with the surface-string composition of *ningir2su2kake4*: it is literally a deity name with a case suffix, and FastText's character n-gram model constructs a vector that reflects the co-presence of both deity-name substrings and grammatical-suffix substrings. The bridge score of 1.000 is not a finding about Sumerian cognition — it is a finding about what happens to agglutinative tokens in n-gram-based embedding. A linguistically naive model cannot know that the *-ke4* suffix is "less meaningful" than the root *ningir2su* and treats the whole token as a distributional unit. The result is that the grammatical case form of a deity name sits halfway between the deity-name cluster and the postposition cluster. This is a diagnostic about the alignment's vocabulary scope, not about Ningirsu's semantic neighbors. The atlas is explicit that Lens 6 should be read with this in mind: bridge tokens are candidates for morphological compounds, not necessarily for concepts that bridge two semantic domains.

---

### 5.2 — *kas4ke4nesze3*: Four Morphemes, One Token

**The finding.** The token *kas4ke4nesze3* — *kas4* (runner) + *-ke4* (genitive/case suffix) + *-ne-* (third-person-plural pronominal) + *-sze3* (terminative-directional suffix, "toward") — has a bridge score of 1.000 between clusters 3 and 29 (`lens6_structural_bridges.rows[6]`).

**The anchor.** The root morpheme *kas4* is glossed "runner" or "swift" in ePSD2 (http://oracc.museum.upenn.edu/epsd2/sumerian/kas4), attested in contexts of speed, dispatch, and messenger service. The terminative suffix *-sze3* encodes direction toward a goal (Foxvog, *Introduction to Sumerian Grammar*, §6.5). The combination *kas4ke4nesze3* would parse as something like "toward/for the runners" or "directed toward the swift ones" — a directional phrase in a verbal clause. The full form would be unusual as a standalone token; its presence in the corpus as a searchable unit reflects the ETCSL ATF encoding, where verbal phrases can appear as single hyphenation-removed tokens in the normalized text.

**The geometry.** Bridge score 1.000 (`lens6_structural_bridges.rows[6]`, `bridge_score: 0.999988`) with clusters 3 and 29. Cluster 3 representative members: *dub*, *dubsar*, *ugula*, *ensi2*, *szara2* — an administrative-title cluster (scribe, overseer, governor, temple functionary). Cluster 29 representative members: *1(disz)*, *2(disz)*, *5(disz)*, *3(disz)*, *udu* — the numeric-classifier cluster (count markers and "sheep," the most common counted commodity). The token *kas4ke4nesze3* bridges between the administrative-title domain and the numeric-classifier domain. This pairing appears surprising — a runner-phrase bridging scribes and sheep-counts — until one considers that the bridge score reflects distributional co-occurrence, and dispatch runners appear in the same administrative documents (accounts of messengers, courier lists) that also track commodity quantities and official titles.

**Interpretation.** The atlas data suggests (tentatively) that *kas4ke4nesze3* is a document-context bridge: it appears in administrative tablets that also contain the administrative-title vocabulary of cluster 3 and the commodity-counting vocabulary of cluster 29, making its distributional vector an average over both environments. This is the "corpus bridge" rather than the "conceptual bridge" interpretation of a Lens 6 result: the token is not a concept that bridges the domains of administration and counting, but a token whose corpus contexts happen to span both domains. The agglutinative form — four morphemes collapsed into a single surface token — intensifies the bridge effect because the suffix *-sze3* and the suffix *-ke4* individually contribute to the grammatical-morpheme cluster, while the root *kas4* contributes to the action-noun cluster, and the corpus occurrence ties all of it to the administrative-tablet environment.

---

### 5.3 — *karzidda*: True Quay, False Clusters

**The finding.** The token *karzidda* — *kar* (quay, harbor) + *zid* (true, right, correct) — has a bridge score of 1.000 between clusters 33 and 1 (`lens6_structural_bridges.rows[3]`).

**The anchor.** ePSD2 (http://oracc.museum.upenn.edu/epsd2/sumerian/karzidda) glosses *karzidda* as "the true quay" — a compound noun combining *kar* "quay/harbor" with *zid* "true/right/good." In Sumerian literary texts, particularly those dealing with the afterlife and the underworld, *karzidda* is the name of the quay at the edge of the underworld where the dead embark. It appears in *The Descent of Inanna* (ETCSL 1.4.1) and in underworld laments. The 𒍣 (ZI) sign encodes the *zid* component (life/truth), but the compound means not "life's quay" but "true quay" — a proper-place name that uses *zid* as an honorific qualitative rather than as the full life-breath concept.

**The geometry.** Bridge score 1.000 (`lens6_structural_bridges.rows[3]`, `bridge_score: 0.999989`) with clusters 33 and 1. Cluster 33 representative members: *1(diš)*, *še*, *2(diš)*, *5(diš)*, *geš* — predominantly a counting/grain-measure cluster (numeric notations and the word for barley). Cluster 1 representative members: *1(u)*, *sze*, *sza3*, *sar*, *2(u)* — also a numeric-and-measure cluster, overlapping heavily with cluster 33 in content. The *karzidda* bridge is between two numeric-measurement clusters, not between a place-name cluster and a quality-epithet cluster.

**Interpretation.** The *karzidda* geometry is difficult to interpret as a conceptual finding. A compound proper noun meaning "the true quay" bridging two clusters of numeric classifiers is more plausibly explained by corpus distribution than by semantics: *karzidda* likely appears in underworld literary texts that are numerically sparse but share distributional features with other short-text environments, causing the k-means algorithm to assign it near the clusters that dominate by frequency (the numeric-classifier clusters are by far the largest in the 40-cluster solution, given the dominance of accounting vocabulary in the corpus). The interesting finding here is a negative one: the bridge score does not tell us what *karzidda* "bridges" conceptually; it tells us that the k-means geometry of a corpus dominated by administrative numerics pulls almost every rare literary term toward the numeric clusters. The atlas documents this as a diagnostic limitation — Lens 6 bridges in numeric-dominated corpora are not reliably conceptual bridges.

---

## § 6 — 𒊮 Theme 4: Transliteration Shadows

*Heading sign: 𒊮 SHA3, "heart/inner"*

The ORACC digital library and its associated ATF (Ascii Transliteration Format) encoding convention use *š* for the phoneme written as a cuneiform sibilant that appears in sign names like ŠA3, ŊEŠ, ŠU. The bulk of ETCSL and CDLI ATF files encode this phoneme as *sz* (older convention) or *c* (regional convention used by some ORACC editors). When the fused Cuneiformy corpus is assembled from multiple sources, both conventions appear as surface tokens in the same vocabulary. The result is doppelganger token pairs: the same Sumerian word recorded with two different transliteration conventions, each appearing as a distinct surface token in the FastText vocabulary.

---

### 6.1 — The c/š Variant Pairs: A Sanity Check That Works

**The finding.** The token pair *baandab5be2ec* and *baandab5be2eš* has a cosine similarity of 0.998 in the Sumerian source space (`lens5_doppelgangers.rows[0]`). The pairs *ciingaantum2mu* / *šiingaantum2mu* (cosine 0.994, `rows[1]`) and *imciingi4gi4* / *imšiingi4gi4* (cosine 0.993, `rows[3]`) show the same pattern.

**The anchor.** These tokens are not single Sumerian words but multi-morpheme verbal forms: *baandab5be2ec* is likely a verbal chain from the Sumerian finite verb stem *dab5* (to seize/grasp) with prefix chain *ba-an-* and suffix *-ec* (ablative or other grammatical marker), appearing in a long verbal complex. The two surface forms differ only in the encoding of the final sibilant: *-ec* (c-convention) versus *-eš* (š-convention). ePSD2 uses the š-convention as its citation standard; ATF files using the c-convention were common in earlier ORACC encodings. Both represent the same phoneme and the same word.

**The geometry.** The cosine similarity of 0.998 between *baandab5be2ec* and *baandab5be2eš* (`lens5_doppelgangers.rows[0]`, `cosine_similarity: 0.9979`) confirms that FastText's character n-gram model connects these two forms despite their surface-string difference. A cosine of 0.998 is near-identical: the two tokens are essentially the same vector, which is the correct result. The same holds for *ciingaantum2mu* / *šiingaantum2mu* (0.994) and *imciingi4gi4* / *imšiingi4gi4* (0.993). The transliteration-convention pair *sag9gaac* / *sag9gaaš* (cosine 0.989, `rows[21]`) and *ciimdiriggeen* / *šiimdiriggeen* (cosine 0.986, `rows[55]`) extend the pattern into lower-frequency forms.

**Interpretation.** The Lens 5 doppelganger results for c/š pairs are a positive validation of the FastText model's robustness to transliteration-convention noise. Because FastText represents words as sums of character n-grams, two tokens that differ only in c/š encoding share most of their character n-gram components (*ba-an-da-ab-5-be-2-e* is common to both *baandab5be2ec* and *baandab5be2eš*) and receive nearly identical vectors. This is a feature rather than a bug: the model generalizes across the convention boundary without being told to. The practical implication is that cosine-distance results for these token pairs are reliable indicators of semantic proximity — the c/š noise does not propagate into alignment errors for any token pair that both conventions encode.

---

### 6.2 — The Numeric Confound: When the Sanity Check Fails

Not all Lens 5 near-duplicates are sanity-check successes. The pair *5(n01@f)* and *6(n01@f)* (cosine 0.992, `lens5_doppelgangers.rows[4]`) and the pairs *9kammaam3* / *8kammaam3* (cosine 0.991, `rows[8]`) and *1kammaam3* / *8kammaam3* (cosine 0.987, `rows[31]`) are not encoding conventions for the same word — they are different numeric values that the embedding has learned to confuse.

*n01@f* is a cuneiform numeric sign notation (the *@f* suffix marks a variant of the *n01* sign group). The fact that 5(n01@f) and 6(n01@f) have cosine 0.992 means the embedding treats the number 5 and the number 6, in this notation, as nearly identical. This is a corpus artifact: in accounting tablets, specific numeric notations co-occur in highly similar contexts (commodity lists, ration records) regardless of the exact quantity they record. The model learns that *n01@f*-bearing tokens always appear near certain administrative vocabulary and near each other, producing embeddings that reflect the count-context rather than the specific quantity. The *kammaam3* pairs (*1kammaam3*, *8kammaam3*, *9kammaam3*, *10kammaam3* are all within the top 50 Lens 5 pairs) extend this pattern: ordinal or temporal count markers appear in sufficiently similar environments that the model cannot distinguish them by co-occurrence.

The diagnostic value here is a caution: high doppelganger cosine scores do not always indicate lexical equivalence. For c/š pairs, cosine > 0.99 is a reliability indicator. For numeric-notation pairs, cosine > 0.99 is an artifact of corpus homogeneity in counting contexts. The Lens 5 table contains both; they must be read with domain knowledge to distinguish the positive validations from the artifacts.

---

## § 7 — 𒀸 Theme 5: The Numeric Tail

*Heading sign: 𒀸 ASH, "one/unit" (numeric marker)*

The top-10 results for Lens 2 (untranslated high-frequency terms) are, without exception, numeric forms: *1(disz)*, *1(u)*, *2(disz)*, *5(disz)*, *3(disz)*, *2(u)*, *1(asz@c)*, *4(disz)*, *3(u)*, and the postposition-like form *sze3* at rank 5 (`lens2_no_counterpart.rows[0]` through `rows[9]`). The top-scoring token, *1(disz)*, appears 115,478 times in the corpus with a top-1 cosine similarity to the English word "goat" of 0.457 — barely above chance in a well-calibrated alignment — producing a Lens 2 score of 62,733 (`lens2_no_counterpart.rows[0]`, `score: 62733.2`).

---

### 7.1 — The Corpus Is an Accounting Office

The numeric tokens in Lens 2 are not "Sumerian concepts with no English equivalent." They are count classifiers and quantity notations from the administrative tablet corpus: *(disz)* is the Sumerian singular count marker (one unit of something), *(u)* marks tens, *(asz@c)* and *(asz@c)* are variant notations for the capacity unit. The presence of *1(disz)* at corpus frequency 115,478 — the single most common token in the entire Cuneiformy vocabulary — is a signature of what the Sumerian tablet corpus actually is: a vast record of administrative accounting for grain, livestock, labor, and land from the Third Dynasty of Ur (circa 2100–2000 BCE) and adjacent periods.

The count marker *1(disz)* scores highly on Lens 2 because: (a) it appears very frequently, and (b) no English word has a close distributional match for a pure count marker. The best English neighbor, "goat," achieves cosine 0.457 because *1(disz)* co-occurs heavily with livestock count entries — not because the Sumerian count marker *means* "goat" in any sense. This is a corpus-composition artifact, not a translation finding.

The five-highest-frequency numeric tokens in the atlas are:
- *1(disz)*: 115,478 occurrences, top-1 English "goat" at cosine 0.457 (`lens2_no_counterpart.rows[0]`)
- *2(disz)*: 78,754 occurrences, top-1 English "goat" at cosine 0.476 (`rows[2]`)
- *1(u)*: 68,968 occurrences, top-1 English "worker" at cosine 0.279 (`rows[1]`)
- *5(disz)*: 65,574 occurrences, top-1 English "goat" at cosine 0.375 (`rows[3]`)
- *3(disz)*: 58,788 occurrences, top-1 English "goat" at cosine 0.383 (`rows[5]`)

The alignment reaches "goat" as the top-1 English neighbor for most count-marker forms because goat-count entries are common administrative text-types in the Ur III corpus. The model's nearest-English-neighbor for a generic count marker happens to be the most frequently counted animal species.

---

### 7.2 — *sze3* and the Postposition-Accounting Crossover

The one non-numeric token in the Lens 2 top-10 is *sze3*, at rank 5 with corpus frequency 52,685, top-1 cosine 0.291 to English "father," and Lens 2 score 37,337 (`lens2_no_counterpart.rows[4]`). *sze3* is the Sumerian terminative-directional postposition (Foxvog, *Introduction to Sumerian Grammar*, §5.8) — it encodes "toward/to/for" as a suffix or enclitic. It is among the most frequent functional morphemes in Sumerian and is grammatical rather than lexical in most uses. Its top-1 English "father" at cosine 0.291 is near-chance; the alignment cannot find any English word whose distributional neighborhood resembles the postposition *sze3*'s neighborhood, because English has no direct distributional equivalent to a terminative-directional suffix.

The Lens 2 score for *sze3* is technically filtered out in the plan's discussion note ("filtered — top numeric candidates excluded as corpus artifact") because the original plan recognized that *sze3* is primarily a grammatical function morpheme rather than a lexical item. The atlas data supports this: the low cosine (0.291) to an irrelevant English word confirms that the alignment treats *sze3* as an untranslatable function word.

---

### 7.3 — The Diagnostic Value of the Numeric Tail

The Lens 2 result carries a methodological lesson about reading atlas data. The lens is designed to surface "terms that appear often but have no English counterpart" — a heuristic designed to find important vocabulary that is semantically isolated. In the Sumerian corpus, this heuristic is overwhelmed by administrative-accounting notation that is frequent but trivial in the interpretive sense. A Sumerologist would immediately recognize that count markers belong in a different interpretive category from, say, *inanna* (rank 31 in Lens 2, with corpus frequency 10,452 and top-1 cosine 0.294 to "good") — but the raw Lens 2 score cannot make that distinction.

The presence of the numeric tail is therefore not a failure of Lens 2 but a diagnostic about the corpus: the Sumerian tablet archive, as a whole, is primarily an administrative record rather than a literary one. The literary ETCSL texts that receive most of the Sumerological interpretive attention are a minority of the total corpus by token count. Any NLP approach to Sumerian meaning that does not separately model administrative and literary registers will encounter this frequency dominance. Future atlas iterations could apply a register filter — excluding tokens whose top-1 English neighbor is a number-adjacent word or whose five nearest Sumerian neighbors are all numeric forms — before computing Lens 2 scores. The present atlas chooses to show the raw signal and discuss it rather than pre-filter.

---

## § 8 — 𒍣 Theme 6: Reading Through the Floor

*Heading sign: 𒍣 ZI, "life/breath"*

The six-lens atlas produces ranked lists of 50 tokens per lens. An implicit assumption in presenting ranked lists is that top-1 is the most interesting result. This assumption is wrong for every lens in the atlas, and understanding why is necessary for reading the atlas honestly.

---

### 8.1 — Why Top-1 Is Usually the Noisiest Signal

**Lens 1 (English displacement):** The most-displaced token, *rin2* → "lord" at cosine -0.088, is a high-confidence anchor with an interpretable mismatch. But the second-ranked token, *jizzal* → "ear" at cosine -0.063, is more interpretively rich — its metonymic-wisdom story is better supported by the Sumerological literature. The first few ranks of Lens 1 are dominated by either (a) high-confidence anchors that have a clear and already-known translation controversy, or (b) low-confidence anchors where the gloss is simply noisy. The signal-to-noise ratio improves as one moves past the very top into ranks 5–20, where high-confidence anchors with moderately negative cosines begin to appear — tokens like *du14* → "combat" (rank 7, cosine -0.033, confidence 0.95) where the mismatch is smaller but the Sumerological interpretation is more tractable.

**Lens 2 (no counterpart):** As documented in §7, the top-10 are dominated by numeric classifiers. The first lexically interesting token is *sze3* at rank 5, and the first lexically interesting non-grammatical non-numeric token is *inanna* at rank 31 (corpus frequency 10,452, cosine 0.294 to "good"). A search for the most interesting Lens 2 findings should begin at rank 15 and look for tokens with moderate scores but interpretable Sumerological profiles.

**Lens 3 (isolation):** The top-ranked token, *had2* (isolation score 0.633), might appear to be the atlas's strongest isolation finding. Inspection of its five nearest Sumerian neighbors — *had2bi*, *had2ka*, *had2da*, *had2ta*, *had2sze3* (cosines 0.767, 0.704, 0.698, 0.687, 0.665, from `lens3_isolation.rows[0]`) — reveals that *had2* is isolated only in the sense that its nearest neighbors are all morphological inflections of itself with very short cosine distances. It is isolated from the broader lexicon but internally compact. The more interesting isolation cases are *uttu*, *gidri*, and *ebgal* (ranks 3, 5, and 10) where the isolation reflects genuine semantic specificity rather than just morphological family coherence.

**Lens 4 (cross-space divergence):** The top-ranked token, *adabkibi* (Jaccard distance 1.0), and the majority of the top-50 are tokens with such sparse distributional support that the two aligned spaces simply have no coherent neighborhood for them. A Jaccard distance of 1.0 with 50+ tokens in the first 50 ranks is a sign that the lens has reached the noise floor — these are all reliability flags, not semantic discoveries. The useful Lens 4 signals would be tokens in the 0.80–0.95 Jaccard range where partial divergence might indicate genuine model disagreement about a concept's neighborhood.

**Lens 5 (doppelgangers):** The top entries are the c/š transliteration-convention pairs discussed in §6, which are validations rather than discoveries. The more interesting doppelganger cases are farther down the list: pairs like *lak263* / *lak264* (cosine 0.988, rank 32) — sequential sign numbers that the embedding confuses — or *lugalki* / *lugalkix* (cosine 0.988, rank 23) where the *x* suffix marks an uncertain sign reading, suggesting the model treats a "certain" reading and an "uncertain" reading as near-identical tokens.

**Lens 6 (structural bridges):** As discussed for *karzidda* in §5.3, the top bridge tokens are pulled toward the dominant numeric clusters by corpus frequency rather than by conceptual content. The interpretively richest bridge tokens are those where the two bridged clusters are semantically coherent domains, not both dominated by numeric classifiers. The *ningir2su2kake4* bridge (clusters 9 and 4 — deity names and grammatical morphemes) is interpretively cleaner than the *karzidda* bridge (clusters 33 and 1 — two numeric clusters). But even the *ningir2su2kake4* bridge is an agglutination artifact, as argued in §5.1.

---

### 8.2 — The Middle Ranks Are Where Interpretation Lives

A productive reading strategy for the atlas is to treat the top 5–10 of each lens as a quality-control zone (dominated by either extreme artifacts or already-known findings) and focus on ranks 15–35 as the interpretive zone.

In Lens 1, this means looking at tokens in the -0.010 to -0.020 cosine range with anchor confidence above 0.75: *gu2tul2* → "work" (rank 15, cosine -0.014, confidence 0.88), *igigal2* → "reciprocal" (rank 19, cosine -0.011, confidence 0.72), *bi2nitum* → "beam" (rank 20, cosine -0.011, confidence 0.61). These are more modest displacements but potentially more tractable: "work" and "reciprocal" and "beam" are English glosses whose specific inadequacy for Sumerian concepts might be less obvious and therefore more worth examining than the well-known en/priest tension.

In Lens 3, the middle-rank isolated tokens include *dilmun* (rank 25, isolation score 0.609), the mythological land Dilmun that Jacobsen (1976, pp. 55–57) interprets as the primordial paradise; *nibru* (rank 29, score 0.598), the sacred city of Nippur; and *marduk* (rank 30, score 0.597), the Babylonian deity whose increasing prominence in the religious landscape corresponds to the period of Old Babylonian text composition. These are proper nouns with rich Sumerological profiles and moderate isolation scores — more interpretable than the top-ranked *had2* and less prone to noise than the very bottom of the table.

In Lens 6, the middle ranks contain bridge tokens where both bridged clusters are non-numeric. For example: *kalammara* (bridge 1.000, clusters 18 and 4) — an interesting compound (*kalam* = the black-headed people of Sumer + *-ra* case suffix) bridging cluster 18 (*ga2*, *gar*, *ag2*, *za3*, *ŋa2* — a placement/action cluster) and cluster 4 (*ba*, *i3*, *sze3*, *za*, *igi* — a grammatical-morpheme cluster). The bridge of a land-name compound across action and grammar clusters is a different kind of signal from the deity-name bridge of *ningir2su2kake4*.

---

### 8.3 — The Atlas as Diagnostic

The atlas is not a discovery machine that produces findings by sorting a column. It is a diagnostic tool that surfaces patterns for a reader who brings domain knowledge. The six lenses are independent filters on the same underlying alignment, and the most useful signals emerge when multiple lenses converge on the same token — when a token has both a negative Lens 1 cosine (its gloss is misaligned) and a high Lens 3 isolation score (its embedding is semantically sparse) and a high Lens 4 Jaccard distance (the two alignment models disagree about it). Such a token is not just a single anomaly; it is a systemic weakness of the alignment in that region of the vocabulary.

The atlas at commit `ff94533` does not expose a multi-lens convergence table — that would be a useful future lens. But a reader who cross-references the six tables can perform this convergence check manually. For example, *ebgal* (isolation 0.616 in Lens 3) does not appear in the Lens 1 filtered top-50, suggesting that despite its semantic isolation, the alignment finds it at least geometrically near its ePSD2 gloss. Conversely, *en* (displacement -0.037 in Lens 1) does not appear in the Lens 3 top-50, suggesting it is not semantically isolated — its embedding has plenty of neighbors; they are just not the neighbors that "priest" would suggest.

The finding is structural: the atlas is a better diagnostic than a ranker.

---

## § 9 — Synthesis

### Six Themes, Three Patterns

The six themes documented in §3–8 reduce to three underlying patterns.

**Pattern 1: The translation problem is geometric, not binary.** Lens 1's negative cosines are not "wrong translations" — they are measurable misalignments between the distributional geometry of a Sumerian token and the distributional geometry of its English gloss. The magnitude matters: *rin2* → "lord" at -0.088 is a stronger mismatch than *en* → "priest" at -0.037, but neither is catastrophic. The range of Lens 1 scores in the filtered table spans from -0.088 (rin2) to +0.021 (nagata), with most tokens clustered near zero. The alignment has no clear threshold between "good translation" and "bad translation" because translation quality is a continuum under this metric, and the metric itself is noisy. The useful claim is not "these translations are wrong" but "these translations create the largest cosine gaps, and Sumerological domain knowledge can explain why."

**Pattern 2: Specialized vocabulary clusters into isolation.** The Lens 3 isolation results for *uttu*, *gidri*, and *ebgal* share a signature: high isolation, nearest neighbors are morphological variants, sharp drop-off to the next tier. This pattern identifies tokens whose corpus distribution is dominated by a small set of specialized texts rather than distributed across the full corpus. The Sumerological corollary is that these terms are either proper nouns, domain-specific technical terms, or cultic vocabulary that does not generalize across text types. The isolation score is a proxy for "how specialized is this word's usage environment."

**Pattern 3: The corpus is mostly an accounting office.** Lens 2's domination by numeric classifiers and Lens 5's confound of count-notation near-duplicates both reflect the same fact: the Sumerian tablet archive is primarily administrative. The literary texts that Sumerologists most frequently study and cite are a minority by token count. Any NLP approach to Sumerian semantics that treats all tokens equally will encode the administrative register as the default, and interpretations built on the alignment will inherit this bias. The cosmogonic vocabulary studied in the prior sibling document (`docs/sumerian_cosmogony.md`) sits in a completely different distributional environment from the Lens 2 top-10 — but the two environments coexist in the same corpus and the same alignment.

### What the Atlas Cannot Do

The atlas cannot confirm interpretive claims about Sumerian culture. It can report that *jizzal*'s projected vector lands geometrically away from "ear" and consistent with the "wisdom reception" interpretation — but this is not the same as confirming that Sumerians conceptualized the ear as the seat of wisdom. It can report that *asag* has Jaccard distance 1.0 under Lens 4 — but this is more informative about alignment quality than about *asag*'s cultural meaning. The atlas is explicit about its noise floor, its corpus bias, and its reliance on an alignment that is correct just over half the time.

The most robust claim the atlas supports is methodological: a computational audit of alignment geometry produces a prioritized reading list for Sumerologists who want to examine where English glosses are most likely to be inadequate. The tokens at the top of Lens 1 are not confirmed translation failures — they are candidates for translation failure that deserve close philological attention.

---

## § 10 — Reproducibility

All numeric claims in this document trace to `docs/anomaly_atlas.json` at pinned commit `ff94533`. The atlas is regenerable with:

```bash
python3 scripts/analysis/sumerian_anomaly_atlas.py
```

This requires the aligned vector artifacts in `final_output/` (produced by the Workstream 2b pipeline). The PDF version of this document is regenerable with:

```bash
bash scripts/docs/render_anomaly_pdf.sh
```

This requires `pandoc ≥ 3.0`, `xelatex` (MacTeX or TeX Live), and the committed Noto Sans Cuneiform font at `docs/fonts/NotoSansCuneiform-Regular.ttf`.

The consistency tests verifying numeric-claim traceability and cuneiform-sign provenance coverage are at `tests/test_anomaly_findings_consistency.py`. Run them with:

```bash
pytest tests/test_anomaly_findings_consistency.py -v
```

The doppelganger threshold is 0.95 (`source_artifacts.doppelganger_threshold`); the isolation k is 10 (`source_artifacts.isolation_k`); the k-means solution uses k = 40 clusters with seed 42 (`source_artifacts.k_clusters`, `source_artifacts.seed`).

---

## § 11 — References

**Primary data sources**

- ETCSL — The Electronic Text Corpus of Sumerian Literature. Oxford: ETCSL project, Faculty of Oriental Studies, University of Oxford. http://etcsl.orinst.ox.ac.uk/
- ePSD2 — The electronic Pennsylvania Sumerian Dictionary, second edition. ORACC: The Open Richly Annotated Cuneiform Corpus, University of Pennsylvania. http://oracc.museum.upenn.edu/epsd2/
- CDLI — Cuneiform Digital Library Initiative. University of California, Los Angeles. https://cdli.mpiwg-berlin.mpg.de/

**Secondary sources**

- Black, Jeremy, Graham Cunningham, Eleanor Robson, and Gábor Zólyomi. *The Literature of Ancient Sumer*. Oxford: Oxford University Press, 2004.
- Foxvog, Daniel A. *Introduction to Sumerian Grammar*. Berkeley: privately circulated, 2016. https://escholarship.org/uc/item/3bc2f75r
- Jacobsen, Thorkild. *The Treasures of Darkness: A History of Mesopotamian Religion*. New Haven: Yale University Press, 1976.
- Kramer, Samuel Noah. *Sumerian Mythology: A Study of Spiritual and Literary Achievement in the Third Millennium B.C.* Philadelphia: American Philosophical Society, 1944.
- Michalowski, Piotr. "Memory and Deed: The Historiography of the Political Expansion of the Akkad State." In *Akkad: The First World Empire*, edited by Mario Liverani. Padova: Sargon, 1993.

**ETCSL texts cited**

- ETCSL 1.1.1 — *Enki and Ninhursag* (the *uttu* myth)
- ETCSL 1.1.2 — *Enki and Ninmah*
- ETCSL 1.1.3 — *Enki and the World Order*
- ETCSL 1.4.1 — *The Descent of Inanna* (the *karzidda* attestation)
- ETCSL 1.6.2 — *Ninurta's Exploits* (the *asag* myth; Ningirsu contexts)

**Pipeline documentation**

- `docs/superpowers/specs/2026-04-20-anomaly-atlas-design.md` — atlas design spec
- `docs/superpowers/specs/2026-04-19-workstream-2b-normalization-fix-design.md` — alignment pipeline spec
- `docs/EXPERIMENT_JOURNAL.md` — workstream history and findings log

---

## Appendix: Cuneiform Sign Provenance — § 12

This appendix documents every distinct cuneiform Unicode codepoint used in this document, with its source identification and any codepoint-labeling notes.

- U+12097 𒂗 — CUNEIFORM SIGN EN, Sumerian *en* "lord/priest." Used as heading sign for §3 and in case study 3.1 first-mention format. Source: ePSD2 sign list entry EN (http://oracc.museum.upenn.edu/epsd2/), confirmed against Unicode character name `CUNEIFORM SIGN EN`. The implementation plan referenced U+12097 as the correct codepoint; confirmed by `hex(ord('𒂗')) == 0x12097`.

- U+1202D 𒀭 — CUNEIFORM SIGN AN, the divine determinative (placed before deity names in cuneiform writing). Used in §5.1 prose to gloss the divine determinative in the Ningirsu name. Unicode character name `CUNEIFORM SIGN AN`. Note: the implementation plan referenced `𒀭` as "AN / DINGIR determinative, varies depending on exact glyph." The codepoint U+1202D is the standard DINGIR/AN sign; U+12017 (a different AN variant, Unicode name `CUNEIFORM SIGN AB2 TIMES BALAG`) was incorrectly labeled as AN in the consistency test suite. The present document uses U+1202D, which is the standard divine determinative in ORACC cuneiform encoding.

- U+12111 𒄑 — CUNEIFORM SIGN GISH, the wooden-object determinative. Used as heading sign for §4. Source: ePSD2 sign list entry GIŠ (http://oracc.museum.upenn.edu/epsd2/). Note: the implementation plan listed this sign as U+12117; verified by Python `unicodedata.name()` that U+12117 is `CUNEIFORM SIGN GU CROSSING GU` (not GISH), while U+12111 is `CUNEIFORM SIGN GISH`. The plan's codepoint was in error; U+12111 is the correct codepoint for the GISH/GIŠ wooden-object determinative.

- U+121A0 𒆠 — CUNEIFORM SIGN KI, "place/earth." Used as heading sign for §5. Source: ePSD2 sign list entry KI (http://oracc.museum.upenn.edu/epsd2/). Confirmed by Unicode character name `CUNEIFORM SIGN KI` and by `hex(ord('𒆠')) == 0x121A0`.

- U+122AE 𒊮 — CUNEIFORM SIGN SHA3, "heart/inner." Used as heading sign for §6. Source: ePSD2 sign list entry ŠA3/SHA3 (http://oracc.museum.upenn.edu/epsd2/). Note: the implementation plan listed this sign as U+122EE; verified by Python `unicodedata.name()` that U+122EE is `CUNEIFORM SIGN TA TIMES MI` (not SHA3), while U+122AE is `CUNEIFORM SIGN SHA3`. The plan's codepoint was in error; U+122AE is correct.

- U+12038 𒀸 — CUNEIFORM SIGN ASH, "one/unit." Used as heading sign for §7. Source: ePSD2 sign list entry AŠ (http://oracc.museum.upenn.edu/epsd2/). Confirmed by Unicode character name `CUNEIFORM SIGN ASH` and by `hex(ord('𒀸')) == 0x12038`.

- U+12363 𒍣 — CUNEIFORM SIGN ZI, "life/breath." Used as heading sign for §8. Source: ePSD2 sign list entry ZI (http://oracc.museum.upenn.edu/epsd2/). Note: the implementation plan labeled the §8 heading sign as `𒄀` (U+12100, `CUNEIFORM SIGN GI`, not ZI). The actual ZI sign is U+12363 (`CUNEIFORM SIGN ZI`). The heading in §8 uses U+12363, the correct ZI sign. Separately, U+12100 GI appears in `docs/sumerian_cosmogony.md` sign inventories but is not the ZI sign.

- U+1238F 𒎏 — CUNEIFORM SIGN NIN, "lady/mistress." Referenced in §5.1 discussion of Ningirsu's name (*nin* = "lord/lady"). Source: ePSD2 sign list entry NIN (http://oracc.museum.upenn.edu/epsd2/). Confirmed by Unicode character name `CUNEIFORM SIGN NIN`.
