# Near-Term Strategy: Geometric Translation of Sumerian Cosmology

## Goal

Use the existing cuneiformy alignment pipeline to surface geometric relationships in Sumerian semantic space that traditional translation flattens — focusing on origin myth, metaphysics, and existential meaning. Ship something surprising within weeks, not months.

## What We Have Right Now

- 3,960 Sumerian words projected into GloVe 300d English space
- Lookup API: `find`, `find_analogy`, `find_blend`
- Anchor pairs connecting Sumerian ↔ English
- The full GloVe English manifold as comparison baseline

This is enough to start finding things.

---

## Phase 1: Concept Cluster Mapping (Week 1-2)

### Pick Three Conceptual Domains

Each domain should be a place where Sumerian cosmology diverges from modern intuition:

**1. Origin / Creation**
- English seed words: `create`, `begin`, `birth`, `origin`, `emerge`, `form`, `earth`, `water`, `heaven`, `separate`
- Sumerian context: Enki, Nammu (primordial sea), the separation of An (heaven) and Ki (earth)
- Question: does the Sumerian cluster for "creation" sit geometrically closer to **water/separation** than to **making/building**? Does creation look more like *division* than *construction*?

**2. Fate / Meaning / Purpose**
- English seed words: `fate`, `destiny`, `purpose`, `decree`, `meaning`, `life`, `death`, `name`, `order`
- Sumerian context: NAM.TAR (fate-cutter), ME (divine decrees/essences of civilization), the Tablet of Destinies
- Question: is "fate" geometrically entangled with "name" in Sumerian space? (In Sumerian thought, to name something is to determine its fate.) Does the cluster structure suggest that meaning and identity are the same concept?

**3. Self / Soul / Consciousness**
- English seed words: `self`, `soul`, `spirit`, `mind`, `heart`, `body`, `breath`, `shadow`, `dream`
- Sumerian context: ZI (life/breath/soul), SA (heart as seat of intention, not emotion)
- Question: does the Sumerian "self" cluster collapse concepts that English separates (mind vs heart vs spirit)? Or does it distinguish things English merges?

### Method

For each domain:

1. **Query the lookup** — `find()` each English seed word, collect the top Sumerian matches
2. **Reverse query** — take those Sumerian words and find their nearest English neighbors. What English words does Sumerian pull *into* the concept that English wouldn't?
3. **Map the cluster geometry** — extract the vectors for all words in the domain (both Sumerian-projected and English-native), compute:
   - Pairwise cosine distances within each cluster
   - Cluster centroid positions
   - Cluster spread (variance)
   - Inter-concept distances (how far is "creation" from "fate" in Sumerian vs English?)
4. **Visualize** — UMAP projection of each domain showing Sumerian words (projected) overlaid on English words, colored by sub-concept

### What to Look For

- **Merges**: concepts that are far apart in English but collapse together in Sumerian (e.g., "name" and "fate")
- **Splits**: a single English concept that fragments into distinct Sumerian clusters (e.g., "soul" → breath-soul vs shadow-soul)
- **Strange attractors**: Sumerian words pulled into a conceptual domain that no English speaker would associate with it
- **Absent regions**: parts of the English cluster with no Sumerian neighbors at all

---

## Phase 2: Geometric Difference Maps (Week 2-3)

### Build a Script: `scripts/geometric_compare.py`

Takes a conceptual domain (list of English seed words) and produces:

1. **Distance matrix comparison** — pairwise distances between concepts in English-native space vs Sumerian-projected space. Output as a heatmap diff. Where the heatmap is hot = Sumerian disagrees with English about how these concepts relate.

2. **Neighborhood divergence** — for each seed word, compare its k-nearest neighbors in English-native vs Sumerian-projected. Jaccard similarity of the neighbor sets = how much Sumerian "agrees" with English about what's near this concept.

3. **Centroid displacement** — the vector from English centroid to Sumerian centroid for each domain. Direction of displacement = which English concepts the Sumerian understanding is pulled *toward*.

4. **Cluster shape comparison** — eigenvalue decomposition of each cluster's covariance matrix. Are the Sumerian clusters more spherical (undifferentiated) or more elongated (structured along a specific axis) than the English ones?

### Output

For each domain, produce:
- A distance matrix diff heatmap (PNG)
- A neighbor divergence table (which concepts diverge most?)
- A 2D UMAP overlay visualization
- A plain-language summary of the most striking geometric differences

---

## Phase 3: Find the Story (Week 3-4)

### Narrative Extraction

Take the most striking geometric findings and trace them back to the source texts:

- If "name" and "fate" cluster together in Sumerian space, find the ETCSL passages where naming and fate co-occur. Show the geometric finding is grounded in textual reality, not embedding noise.
- If creation clusters with water/separation rather than making/building, point to Enuma Elish / Sumerian creation narratives where the cosmogony is literally about waters dividing.
- If the "self" region is shaped differently, connect it to what we know about Sumerian conceptions of personhood.

### The Deliverable

A short write-up (or notebook) that presents 3-5 findings in the format:

> **Traditional translation says:** "nam.tar" means "fate"
>
> **Geometric translation reveals:** "nam.tar" sits at the intersection of cutting, naming, and cosmic order — equidistant from concepts that English treats as completely unrelated. The Sumerian geometric structure suggests fate is not something that *happens to you* but something that is *spoken into you* through the act of naming, which is itself an act of cosmic division.
>
> **Visualization:** [cluster map showing the geometric relationship]
>
> **Source texts:** [ETCSL references]

---

## Success Criteria

We've found something worth publishing/sharing if any of the following are true:

- [ ] A geometric relationship reveals a conceptual structure that Sumerologists recognize as valid but that doesn't come through in standard translation
- [ ] The distance matrix diff shows a statistically significant structural difference between Sumerian and English in at least one domain
- [ ] A "strange attractor" surfaces a connection between concepts that produces a genuine insight about Sumerian metaphysics
- [ ] The findings hold up when cross-referenced against ETCSL source texts

## Tools Needed

- numpy, scipy (already in requirements.txt)
- matplotlib or plotly (for heatmaps and cluster viz)
- umap-learn (for 2D projections)
- The existing `SumerianLookup` class + GloVe vectors

## Risks

- **11% Top-1 accuracy** — the alignment is noisy. Geometric findings need to be robust to this noise. Mitigation: focus on cluster-level patterns rather than individual word positions; use bootstrap resampling to test stability.
- **Corpus bias** — ETCSL is heavily literary/mythological. The embeddings over-represent royal hymns and creation myths. For this particular study, that's actually a feature not a bug.
- **Projection artifacts** — aligning into GloVe space may impose English structure on Sumerian. Mitigation: compare findings against native Sumerian FastText space (pre-alignment) to verify the structure isn't an artifact of the projection.
