# Research Vision: Riemannian Geometry of Affective Semantic Space Across Civilizational Time

## Thesis

By comparing the geometric structure of embedding spaces derived from ancient languages (Sumerian, Egyptian hieroglyphics) against modern English, we can measure how the human capacity for meaning-making has transformed over 4,000+ years — not through subjective interpretation, but through the differential geometry of semantic manifolds.

The shape of a language's embedding space is a crude but measurable model of the mind that produced it. If that shape has changed, the mind has changed. The nature of that change — rotation, compression, expansion, collapse — tells us *what* changed and *what was lost*.

## Core Concepts

### The Embedding Manifold as Cognitive Fossil Record

Each language's embedding space defines a high-dimensional manifold where proximity encodes meaning. The topology of that manifold — its curvature, density, clustering, and voids — reflects how speakers of that language structured their experience of reality.

- **Regions of high curvature** = fine-grained conceptual distinctions (many nearby but distinct meanings)
- **Flat regions** = coarse or undifferentiated conceptual territory
- **Voids** = concepts the language cannot articulate fluently
- **Density** = how much cognitive attention a culture pays to a domain

### What the Ancient-Modern Comparison Would Reveal

Comparing the Sumerian and English manifolds on overlapping conceptual domains (agriculture, kinship, death, authority, the divine, causation, time, self) should yield three measurable objects:

1. **The Axis of Rotation** — concepts that occupy roughly the same geometric position across both manifolds. These are candidates for human universals: the invariants of meaning that persist across all cultural transformation.

2. **The Gap** — regions of rich curvature in the ancient manifold that have flattened or collapsed in the modern one (and vice versa). This is the measurable signature of cognitive loss and gain.

3. **The Polarity** — the vector defined by the direction of rotation from ancient to modern. This describes the *trajectory* of human consciousness as a mathematical object, derived from language geometry rather than ideology.

### Hypothesized Losses (Testable)

Regions where the ancient manifold may show richer structure than the modern:

- **Relational obligation** — Sumerian encodes dozens of social reciprocity concepts; modern English has "duty" and "obligation" and little else
- **Sacred granularity** — ancient languages differentiate many qualities of divine presence where modern language collapses to "spiritual" vs "secular"
- **Embodied cognition** — ancient root words are overwhelmingly somatic; modern abstractions have drifted geometrically distant from bodily experience
- **Temporal experience** — non-linear, non-progressive time structures that modern tense systems have overwritten

### Hypothesized Gains (Testable)

- **Interiority / selfhood** — the conceptual region around individual identity, autonomy, and subjective experience
- **Mechanical causation** — cause-and-effect as distinct from divine will
- **Abstract reasoning** — mathematical, logical, and scientific concept-space

## Method (High-Level)

### Phase 1: Foundation (Current Work)

Cuneiformy already provides the critical first piece:

- Sumerian transliterated corpus (ETCSL, CDLI, ORACC) cleaned and tokenized
- FastText 768d embeddings trained on Sumerian
- Ridge regression alignment into GloVe 300d English space
- Anchor pairs from ePSD2 dictionary and ETCSL parallel translations
- Current accuracy: 11.24% Top-1 (baseline, with room for improvement)

The sister project heiroglyphy provides the Egyptian baseline at 32.35% Top-1.

### Phase 2: Geometric Analysis

- Compute curvature, density, and topological features of the Sumerian embedding manifold
- Compute the same for the English GloVe manifold
- Identify the **diffeomorphism** (mapping function) between them on shared anchor concepts
- Measure the **distortion tensor** of that mapping — where does it stretch, compress, rotate?

### Phase 3: Comparative Baselines

Add intermediate points along the historical timeline to plot the curve of rotation:

- Akkadian (Semitic, overlapping with Sumerian corpus period)
- Classical Greek (Homer → Plato spans the emergence of interiority)
- Latin
- Old English → Modern English

Each additional language provides a data point on the trajectory. Is the rotation smooth? Accelerating? Did it lurch at specific historical moments?

### Phase 4: The Polarity

- Extract the principal axis of rotation from the trajectory curve
- Characterize the gap regions (lost vs gained structure)
- Define the polarity vector: the direction humanity has been moving, expressed as semantic geometry
- Test whether the polarity correlates with known civilizational inflection points (invention of writing, monotheism, printing press, internet)

## Connection to Affective Computing

Recent work in affective computing shows that embedding geometry measurably deforms under emotional state:

- Psychiatric speech analysis shows semantic space **compression** in depression
- Emotion topology research extracts fundamental emotional components from embedding geometry
- LLMs develop well-defined internal emotion geometry at specific transformer depths

The same Riemannian framework that measures ancient-modern drift could measure mood-driven deformation within a single language — treating affect as a local curvature perturbation on the semantic manifold.

This connects two questions:
1. How has the shape of human meaning changed over millennia? (civilizational scale)
2. How does the shape of human meaning change moment-to-moment with emotional state? (individual scale)

If both produce measurable curvature, the relationship between them becomes a question: is individual emotional variation a local fluctuation on a manifold that is itself drifting over civilizational time?

## Tools and Dependencies

- **Embedding training**: gensim (FastText), GloVe, potentially transformer-based models for modern languages
- **Riemannian geometry**: geomstats, scikit-learn manifold learning, custom curvature estimation
- **Visualization**: UMAP/t-SNE for projections, matplotlib/plotly for curvature maps
- **Ancient language corpora**: ETCSL, CDLI, ORACC (Sumerian); Thesaurus Linguae Aegyptiae (Egyptian); Perseus Digital Library (Greek/Latin)

## Open Questions

- Can embedding geometry trained on fragmentary ancient corpora (~40MB for Sumerian) produce reliable manifold structure, or is the corpus too sparse?
- Is the alignment layer (Ridge regression into GloVe space) itself distorting the geometry we want to measure? Should we compare manifolds in their native spaces instead?
- How do we disentangle "this concept didn't exist" from "this concept existed but the corpus doesn't preserve it"?
- What is the right null model — how much geometric distortion do you get just from corpus size differences?
