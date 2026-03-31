# Geometric Translation Pipeline + LaTeX Narrative

## Overview

A three-stage pipeline that performs geometric comparison of Sumerian and English embedding spaces, extracts parallel passages with "geometric translations," and generates a long-form philosophical essay as a LaTeX document. The essay weaves geometric findings into a meditation on how human meaning-making has shifted over 4,000 years, with embedded visualizations and three-column parallel text presentations.

## System Architecture

```
geometric_compare.py → findings.json + figures/
        ↓
geometric_translate.py → parallel passages (transliteration, standard, geometric)
        ↓
generate_narrative.py → LaTeX document (essay + embedded figures + parallel texts)
        ↓
pdflatex → final PDF
```

## Script 1: Geometric Comparison (`scripts/geometric_compare.py`)

### Input

- Aligned Sumerian vectors (`final_output/sumerian_aligned_vectors.npz`)
- Sumerian vocab (`final_output/sumerian_aligned_vocab.pkl`)
- GloVe vectors (`data/processed/glove.6B.300d.txt`)
- Concept domain definitions (hardcoded or YAML config)

### Concept Domains

Three domains, each defined by English seed words:

**Creation / Origin:**
`create`, `begin`, `birth`, `origin`, `emerge`, `form`, `earth`, `water`, `heaven`, `separate`, `divide`, `first`, `primordial`, `chaos`, `order`

**Fate / Meaning / Purpose:**
`fate`, `destiny`, `purpose`, `decree`, `meaning`, `life`, `death`, `name`, `order`, `tablet`, `judge`, `decide`, `law`, `divine`

**Self / Soul / Consciousness:**
`self`, `soul`, `spirit`, `mind`, `heart`, `body`, `breath`, `shadow`, `dream`, `blood`, `eye`, `inner`, `thought`, `will`

### Analysis Per Domain

1. **Cluster extraction**
   - For each English seed word, query `find()` to get top-k Sumerian neighbors
   - Reverse-query: take those Sumerian words, find their nearest English neighbors via cosine similarity against GloVe
   - Collect the expanded word set (seed words + discovered English words + Sumerian words)

2. **Distance matrix diff**
   - Compute pairwise cosine distances between all seed words in English-native GloVe space
   - Compute pairwise cosine distances between the same concepts in Sumerian-projected space (using the Sumerian word closest to each English seed as the proxy)
   - Output: difference heatmap (where hot = Sumerian disagrees with English about the relationship between these concepts)

3. **Neighborhood divergence**
   - For each seed word, get k=20 nearest neighbors in English-native space and in Sumerian-projected space
   - Compute Jaccard similarity of the two neighbor sets
   - Low Jaccard = high divergence = interesting

4. **Centroid displacement**
   - Compute centroid of the domain cluster in English-native space
   - Compute centroid of the corresponding Sumerian-projected cluster
   - The displacement vector, projected back into interpretable English directions (nearest English words to the displacement vector), reveals which concepts Sumerian pulls the domain toward

5. **Cluster shape analysis**
   - Eigenvalue decomposition of covariance matrix for each domain cluster in both spaces
   - Compare: ratio of eigenvalues (spherical vs elongated), effective dimensionality
   - A more elongated Sumerian cluster = the concept is structured along a specific axis that English treats as uniform

6. **Merge/split detection**
   - Identify pairs of seed words where:
     - English distance is large but Sumerian distance is small → **merge** (Sumerian treats these as the same concept)
     - English distance is small but Sumerian distance is large → **split** (Sumerian distinguishes what English conflates)
   - Threshold: distance ratio > 2x flags a candidate

### Output

- `results/geometric_findings.json` — all numerical results, organized by domain
- `results/figures/` — PNG visualizations:
  - `{domain}_distance_diff.png` — heatmap diff
  - `{domain}_umap.png` — 2D UMAP projection with Sumerian (projected) and English words overlaid, colored by sub-concept
  - `domain_comparison.png` — summary visualization comparing all three domains

## Script 2: Geometric Translation (`scripts/geometric_translate.py`)

### Input

- `results/geometric_findings.json` (from step 1)
- ETCSL source texts (`data/raw/etcsl_texts.json`)
- Sumerian aligned vectors + GloVe vectors (for neighbor lookups)

### Process

For each significant finding (merges, splits, strange attractors) from the geometric analysis:

1. **Source text search**
   - Identify the Sumerian terms involved in the finding
   - Search ETCSL texts for lines containing co-occurrences (or individual occurrences in relevant compositions — creation myths for creation domain, etc.)
   - Select the most illustrative passages (up to 3 per finding)

2. **Standard translation extraction**
   - Pull the existing English translation from ETCSL parallel texts

3. **Geometric gloss generation**
   - For each Sumerian word in the passage, look up its k=5 nearest English neighbors in the aligned space
   - Construct a "geometric translation" that uses the embedding neighborhood as the dictionary:
     - If ePSD says "nam.tar" = "fate" but the geometry places it equidistant from "cutting," "naming," and "cosmic decree," the geometric translation reads something like "the naming-cut / the decree-that-severs"
   - The geometric translation is a *literal reading of the geometry*, not a literary interpretation
   - Include the evidence: for each reinterpreted word, list the top-5 English neighbors and their cosine similarities

### Output

- `results/parallel_passages.json` — array of passage objects:
  ```json
  {
    "finding_id": "creation_merge_water_origin",
    "domain": "creation",
    "source": "ETCSL 1.1.1 (Enki and Ninhursag)",
    "line_ref": "c.1.1.1.1-5",
    "transliteration": "...",
    "standard_translation": "...",
    "geometric_translation": "...",
    "word_evidence": {
      "nam": {"top_neighbors": [["name", 0.72], ["identity", 0.68], ...], "epsd_gloss": "fate, lot"},
      ...
    }
  }
  ```

## Script 3: Narrative Generator (`scripts/generate_narrative.py`)

### Input

- `results/geometric_findings.json`
- `results/parallel_passages.json`
- `results/figures/*.png`

### Process

1. **Domain ranking** — score each domain by geometric interestingness (number of significant merges/splits, magnitude of centroid displacement, neighborhood divergence). The narrative leads with the most interesting domain.

2. **Narrative assembly** — produce LaTeX content structured as:

   **Opening (1-2 pages)**
   - The premise: what if translation isn't just wrong but *dimensionally impoverished*?
   - What embedding geometry is, for a non-technical reader
   - What "geometric translation" means — reading the shape of meaning rather than swapping words

   **Per-domain findings (3-5 pages each, for top domains)**
   - The geometric finding, stated plainly
   - Visualization (UMAP cluster map and/or distance heatmap diff)
   - A parallel passage presentation:
     - Column 1: Sumerian transliteration
     - Column 2: Standard English translation
     - Column 3: Geometric translation
   - What the geometry reveals that the standard translation flattens
   - Connection to what we know about Sumerian thought/cosmology

   **Synthesis (2-3 pages)**
   - What the findings collectively suggest: merges reveal what we've separated, splits reveal what we've collapsed
   - The rotation metaphor — the manifold hasn't shrunk, it's reoriented
   - What might be lost (regions of ancient curvature that have flattened)

   **Closing (1 page)**
   - The polarity question — the vector of civilizational drift as a mathematical object
   - Gesture toward the longer research vision (more languages, the trajectory curve)
   - Final parallel passage — the most striking one, presented without commentary

3. **LaTeX generation** — output a complete, compilable `.tex` file

### LaTeX Document Structure

```latex
\documentclass[11pt, oneside]{memoir}
\usepackage{graphicx, xcolor, fontspec, microtype}
\usepackage{paracol}  % for parallel column passages
\usepackage[margin=1.2in]{geometry}

% Custom environment for parallel passages
\newenvironment{parallelpassage}{...}{...}
% Three columns: transliteration | standard | geometric

% Minimal headers — essay style, not academic
% No abstract, no numbered sections
% Section breaks via whitespace and ornamental dividers
```

- Font: a readable serif (TeX Gyre Pagella or similar)
- Figures: full-width, with captions that contribute to the narrative (not just labels)
- Parallel passages: three-column layout using `paracol`, with subtle background tinting to distinguish the geometric translation column
- No footnotes — all evidence inline or in the word evidence appendix
- Optional appendix: full word evidence tables (neighbor lists + similarities) for readers who want to verify the geometric claims

### Output

```
output/
├── geometric_narrative.tex
├── geometric_narrative.pdf  (if pdflatex available)
└── references.bib
```

## Dependencies

Add to `requirements.txt`:
- `umap-learn` — 2D manifold projections
- `matplotlib` — figures and heatmaps
- `seaborn` — heatmap styling

Already present: `numpy`, `scipy`, `scikit-learn`, `gensim`

LaTeX compilation requires a TeX distribution (MacTeX / texlive) with `xelatex` for fontspec support. The pipeline should still produce the `.tex` file even if LaTeX is not installed.

## Concept Domain Definitions

Stored as a dict in `geometric_compare.py` (no external config needed for three domains):

```python
DOMAINS = {
    "creation": {
        "name": "Creation & Origin",
        "seeds": ["create", "begin", "birth", "origin", "emerge", "form",
                  "earth", "water", "heaven", "separate", "divide", "first",
                  "primordial", "chaos", "order"],
        "etcsl_compositions": ["1.1.1", "1.1.2", "1.7.1"],  # creation narratives
    },
    "fate": {
        "name": "Fate, Meaning & Purpose",
        "seeds": ["fate", "destiny", "purpose", "decree", "meaning", "life",
                  "death", "name", "order", "tablet", "judge", "decide",
                  "law", "divine"],
        "etcsl_compositions": ["1.1.4", "1.3.1", "4.32.1"],
    },
    "self": {
        "name": "Self, Soul & Consciousness",
        "seeds": ["self", "soul", "spirit", "mind", "heart", "body",
                  "breath", "shadow", "dream", "blood", "eye", "inner",
                  "thought", "will"],
        "etcsl_compositions": ["1.8.1.4", "2.1.1", "1.4.1"],
    },
}
```

## Error Handling

- If a seed word has no Sumerian neighbors above a minimum similarity threshold (0.1), exclude it from that domain's analysis and note the gap
- If ETCSL text search finds no passages for a finding, note it as "geometrically attested but textually ungrounded" — still include the geometric finding, flag the absence
- If fewer than 2 domains produce interesting findings, the narrative focuses on the one that works rather than forcing thin material

## Success Criteria

The pipeline has produced something worth sharing if:

- At least one domain shows statistically significant geometric divergence (permutation test, p < 0.05)
- At least one parallel passage produces a geometric translation that reveals a non-obvious conceptual relationship
- The LaTeX document compiles to a readable essay that a non-specialist could follow
- The geometric claims are grounded: each reinterpreted word includes its neighbor evidence
