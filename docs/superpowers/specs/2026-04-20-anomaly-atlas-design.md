# Sumerian Anomaly Atlas (+ civilization-agnostic framework)

**Date:** 2026-04-20
**Status:** Approved (brainstorming), pending writing-plans
**Branch:** `master` (to be cut into a fresh feature branch at implementation time)
**Follows:** `docs/superpowers/specs/2026-04-19-sumerian-cosmogony-document-design.md`
**Journal:** `docs/EXPERIMENT_JOURNAL.md`

## Summary

Produce `docs/anomaly_atlas.json` + `docs/anomalies/*.md`: a diagnostic-only atlas applying six anomaly lenses to the 35,508-word whitened-Gemma-aligned Sumerian vocabulary. The six lenses surface (1) anchor pairs where the Sumerian projection lands geometrically far from its English gloss; (2) high-corpus-frequency Sumerian words with no close English counterpart; (3) isolated Sumerian words with distant nearest neighbors; (4) Sumerian words whose Gemma and GloVe top-K neighbors diverge; (5) near-identical Sumerian word pairs (`cos ≥ 0.95`); (6) structural-bridge Sumerian words equidistant from multiple semantic clusters.

Built as a civilization-agnostic framework: the pure-function lens module and the configurable runner are portable to any future aligned-embedding pipeline (Egyptian via a Gemma-tized Heiroglyphy, or future Akkadian/Greek/Latin). Sumerian is the first consumer; the framework is designed for export to the queued comparative repo from day one.

## Motivation

The Sumerian cosmogony document drew interpretive claims from five hand-picked concepts. The atlas tests those claims at scale: does the pattern "Sumerian cosmogonic vocabulary is geometrically distinct from English" actually hold across the 35,508-word vocab, or was it an artifact of the five chosen concepts? If the pattern holds, the atlas surfaces *which other* concepts exhibit it — i.e., material for follow-up writing. If the pattern doesn't hold, the cosmogony document's thesis needs qualification.

Separately, the atlas is a data-generation step for the queued comparative-civilization work. The same six lenses applied to Egyptian would produce immediately-comparable rankings — "here are Sumerian's most-isolated concepts; here are Egyptian's; do the lists overlap on universals or diverge on cultural particulars?" That comparison requires the atlases to exist first, in a shared output shape.

## Audience

This workstream produces a DIAGNOSTIC artifact — not a published document. The consumers are:
- The human researcher (you) — scanning rankings to surface interesting anomalies for future writing.
- Future workstreams — the comparative repo imports the framework as a library and produces an Egyptian atlas.
- Future research — the atlas JSON is a data artifact others can build on.

Per-lens markdown files include methodology framing (what the lens measures, how to read the table) but NO interpretive prose on specific findings. Interpretation happens in future documents that cite the atlas.

## Scope

### In scope

- New `scripts/analysis/anomaly_lenses.py` — six pure functions, civilization-agnostic, unit-testable against synthetic inputs.
- New `scripts/analysis/anomaly_framework.py` — `AnomalyConfig` dataclass + shared `run_atlas` orchestrator.
- New `scripts/analysis/sumerian_anomaly_atlas.py` — thin Sumerian-specific wrapper that builds the config and calls the framework.
- New `tests/analysis/test_anomaly_lenses.py` — unit tests for each lens.
- Atlas outputs:
  - `docs/anomaly_atlas.json` — canonical numeric atlas (schema version 1).
  - `docs/anomalies/atlas_summary.md` — ~200-word index.
  - `docs/anomalies/lens1_english_displacement.md` through `lens6_structural_bridges.md` — per-lens tables + framing prose (no interpretation).
- Optional two histogram PNGs (Lens 3 isolation distribution, Lens 5 doppelganger similarity distribution) if ASCII representation is insufficient.
- Journal entry upon completion.

### Out of scope

- Interpretive prose on specific findings (promoted to a follow-up document if rankings are rich enough).
- Cross-civilizational comparison (blocked on the comparative repo existing and Heiroglyphy being Gemma-tized).
- Alignment pipeline changes.
- Anchor semantic-quality remediation (atlas surfaces noise via Lens 1 unfiltered + Lens 4; removing noisy anchors is a separate workstream).
- Interactive visualizations.
- UMAP-based clustering for Lens 6 (k-means on 768d is simpler and deterministic).
- Any library-packaging work (no `setup.py`, no PyPI). The framework is *internally* portable via sibling-directory imports, not via pip install.

### Deliverables produced

- 3 new scripts, 1 new test file, 1 atlas JSON, 7 markdown files, 1 journal entry. Optional 2 PNGs.

## Success Criteria

- `docs/anomaly_atlas.json` regenerates byte-identically from `sumerian_anomaly_atlas.py` (deterministic seeds + sorted iteration).
- All six lenses produce non-empty ranked tables in the JSON.
- All seven markdown files committed with top-50 tables (or histograms for Lens 3 / Lens 5).
- `pytest tests/analysis/test_anomaly_lenses.py` — all tests pass.
- `pytest tests/ --ignore=tests/test_integration.py` — no regressions in existing 144 tests.
- Atlas summary markdown has one data-driven sentence per lens describing the top-1 row (e.g., "Lens 1 top displacement: `X` ↔ `Y`, cosine Z").
- The `anomaly_framework.AnomalyConfig` dataclass + `run_atlas` function are cleanly importable by a sibling-repo orchestrator (no Sumerian-specific constants in the framework module).
- Atlas generation runtime <10 min on the reference laptop.

## Design

### Architecture and file layout

```
scripts/analysis/
  anomaly_lenses.py                NEW — 6 pure functions, civilization-agnostic
  anomaly_framework.py             NEW — AnomalyConfig + run_atlas
  sumerian_anomaly_atlas.py        NEW — Sumerian-specific orchestrator wrapper
  (existing files unchanged)

tests/analysis/
  test_anomaly_lenses.py           NEW — synthetic-input tests for all 6 lenses

docs/
  anomaly_atlas.json               NEW — canonical numeric atlas
  anomalies/                       NEW directory
    atlas_summary.md               short index + one-sentence-per-lens
    lens1_english_displacement.md
    lens2_no_counterpart.md
    lens3_isolation.md
    lens4_cross_space_divergence.md
    lens5_doppelgangers.md
    lens6_structural_bridges.md

docs/figures/anomalies/             NEW directory (only if PNGs needed)
  lens3_isolation_histogram.png     optional
  lens5_doppelganger_histogram.png  optional
```

### Framework module (`anomaly_framework.py`)

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class AnomalyConfig:
    civilization_name: str                    # e.g., "sumerian", "egyptian"
    aligned_gemma_path: Path                  # source language aligned into Gemma
    aligned_glove_path: Path | None           # source aligned into GloVe; optional (Lens 4 requires both)
    source_vocab_path: Path                   # source-language vocab (same order as aligned vectors)
    target_gemma_vocab_path: Path             # e.g. english_gemma_whitened_768d.npz
    target_glove_vocab_path: Path | None      # e.g. glove.6B.300d.txt
    anchors_path: Path                        # anchor pairs JSON
    corpus_frequency_path: Path               # cleaned_corpus.txt or equivalent
    junk_target_glosses: frozenset[str]       # language-specific junk filter
    min_anchor_confidence: float              # filter threshold for Lens 1 filtered tier
    min_token_length: int                     # source-side length filter
    output_atlas_json: Path
    output_markdown_dir: Path
    output_figures_dir: Path | None           # None = use ASCII histograms
    seed: int = 42
    k_clusters: int = 40                      # Lens 6 k-means k
    top_n_per_lens: int = 50
    doppelganger_threshold: float = 0.95
    isolation_k: int = 10

def run_atlas(config: AnomalyConfig) -> dict:
    """Orchestrate the 6 lenses, render JSON + markdown, return summary dict."""
```

`run_atlas` loads artifacts based on config paths, calls each lens function in sequence, renders outputs. Lens 4 gracefully skips with a documented note in the markdown if `aligned_glove_path` is None.

### Lens module (`anomaly_lenses.py`)

Six pure functions, no I/O, no hardcoded paths. Each takes numpy arrays + lookup maps + threshold parameters, returns a list of ranked row-dicts. Signatures (final; stable across civilizations):

```python
def lens1_english_displacement(
    aligned_gemma: np.ndarray,              # (N_source, 768)
    source_vocab: list[str],
    target_gemma_vectors: np.ndarray,       # (N_target, 768)
    target_gemma_vocab: dict[str, int],
    anchors: list[dict],                    # [{sumerian, english, confidence, source}, ...]
    top_n: int,
    junk_target_glosses: frozenset[str],
    min_token_length: int,
    min_anchor_confidence: float,
) -> dict:
    """Return {'rows_unfiltered': [...], 'rows_filtered': [...], 'filter_rules_applied': [...]}"""

def lens2_no_counterpart(
    aligned_gemma: np.ndarray,
    source_vocab: list[str],
    anchor_source_tokens: frozenset[str],   # pre-computed set of source tokens in the anchor pool
    target_gemma_vectors: np.ndarray,
    target_gemma_vocab: list[str],          # reverse mapping
    corpus_frequency: dict[str, int],
    top_n: int,
) -> dict:
    """Return {'rows': [{sumerian, corpus_frequency, top1_english, top1_cosine, score}, ...]}"""

def lens3_isolation(
    aligned_gemma: np.ndarray,
    source_vocab: list[str],
    isolation_k: int,
    top_n: int,
) -> dict:
    """Return {'rows': [...], 'histogram': {bin_edges, counts}}. Each row has {sumerian, distance_to_kth_neighbor, nearest_5_neighbors}."""

def lens4_cross_space_divergence(
    aligned_gemma: np.ndarray,
    aligned_glove: np.ndarray,
    source_vocab: list[str],
    anchor_source_tokens: frozenset[str],
    top_n: int,
    neighbors_k: int = 10,
) -> dict:
    """Return {'rows_unfiltered': [...], 'rows_anchor_only': [...]} with Jaccard distance between Gemma and GloVe top-K sets."""

def lens5_doppelgangers(
    aligned_gemma: np.ndarray,
    source_vocab: list[str],
    anchor_source_tokens: frozenset[str],
    threshold: float,
    top_n: int,
    chunk_size: int = 500,
) -> dict:
    """Return {'rows': [...], 'histogram': {bin_edges, counts}}. Each row has {sumerian_a, sumerian_b, cosine_similarity, in_anchor_set: (bool, bool)}."""

def lens6_structural_bridges(
    aligned_gemma: np.ndarray,
    source_vocab: list[str],
    k_clusters: int,
    top_n: int,
    seed: int,
) -> dict:
    """Run k-means; compute bridge score per token; return {'k_clusters', 'rows': [...]} where each row has {sumerian, bridge_score, nearest_cluster, second_nearest_cluster, cluster_A_members, cluster_B_members}."""
```

All functions use pre-normalized vectors (caller's responsibility to pass L2-normalized). Deterministic with fixed seed.

### Atlas JSON schema (`docs/anomaly_atlas.json`)

```json
{
  "atlas_schema_version": 1,
  "atlas_date": "YYYY-MM-DD",
  "civilization": "sumerian",
  "source_artifacts": {
    "aligned_gemma_path": "...",
    "aligned_glove_path": "...",
    "source_vocab_path": "...",
    "anchors_path": "...",
    "anchors_sha256": "...",
    "corpus_frequency_path": "...",
    "corpus_frequency_sha256": "...",
    "seed": 42,
    "k_clusters": 40,
    "top_n_per_lens": 50,
    "doppelganger_threshold": 0.95,
    "isolation_k": 10
  },
  "summary": {
    "total_aligned_tokens": 35508,
    "anchor_tokens_in_vocab": 8998,
    "non_anchor_tokens_in_vocab": 26510,
    "top1_per_lens": {
      "lens1_english_displacement": "<sumerian> -> <english> (cos=<x>)",
      "lens2_no_counterpart": "<sumerian> (freq=<n>, top1_cos=<x>)",
      "lens3_isolation": "<sumerian> (d_10=<x>)",
      "lens4_cross_space_divergence": "<sumerian> (jaccard=<x>)",
      "lens5_doppelgangers": "<a> == <b> (cos=<x>)",
      "lens6_structural_bridges": "<sumerian> (bridge=<x>, clusters <A>/<B>)"
    }
  },
  "lens1_english_displacement": { ... },
  "lens2_no_counterpart":       { ... },
  "lens3_isolation":            { ... },
  "lens4_cross_space_divergence": { ... },
  "lens5_doppelgangers":        { ... },
  "lens6_structural_bridges":   { ... }
}
```

### Per-lens markdown format

Each `docs/anomalies/lens<N>_<name>.md` has:
- ~100-word framing paragraph: what the lens measures, what would be noise vs signal.
- Top-50 table (or both tiers for Lens 1 + Lens 4).
- For Lens 3 and Lens 5: histogram (ASCII or PNG reference).
- Cross-reference footer: pointer back to `docs/anomaly_atlas.json` for full data, and to `docs/sumerian_cosmogony.md` if any of the concepts from the cosmogony's 5 deep dives appear in the top-50.

`docs/anomalies/atlas_summary.md` has:
- Short intro (~80 words): what the atlas is, methodology provenance, pinned commit of alignment artifacts.
- Per-lens section with one data-driven sentence ("Lens 1: top displacement outlier is `X` → `Y` at cosine Z, see [lens1 markdown](lens1_english_displacement.md)").

### Determinism

- All lenses use seeded operations: k-means with `random_state=seed`; chunked cosine computations iterate in vocab order (already sorted).
- JSON written with `sort_keys=True, indent=2`.
- Markdown rendering is deterministic given JSON contents.
- Rerun produces byte-identical JSON; markdown output likewise.

## Error Handling

| Condition | Behavior |
|---|---|
| `aligned_glove_path` is None | Lens 4 skipped; markdown reports "skipped (requires dual-target alignment)" |
| Source vocab and aligned vectors have mismatched row counts | `ValueError` |
| Corpus frequency file missing | Lens 2 runs with frequency=0 for all tokens; markdown notes degraded mode |
| Anchors JSON malformed or missing required keys | `ValueError` |
| k-means fails to converge within default iterations | Log warning, retry with doubled iteration count |
| Any lens function raises in production run | Catch, log, emit empty lens section in JSON with `"error": "..."` field, continue with remaining lenses |

## Testing Strategy

New `tests/analysis/test_anomaly_lenses.py`. Tiny synthetic inputs; target <2s total runtime.

### Per-lens unit tests

- `test_lens1_ranks_low_cosine_first` — synthetic anchors with known displacement; top row should have smallest cosine.
- `test_lens1_filtered_excludes_short_english` — junk gloss like "c" appears in unfiltered but not filtered tier.
- `test_lens2_score_combines_frequency_and_low_top1` — two tokens with known frequency and top-1 cosine values produce expected ranking.
- `test_lens3_isolation_is_k_nearest_distance` — for a synthetic 5-token vocab with known geometry, Lens 3 returns the distance to the kth neighbor.
- `test_lens4_jaccard_distance_all_different` — synthetic top-K lists that share no words produce Jaccard=1.
- `test_lens4_jaccard_distance_all_same` — identical top-K lists produce Jaccard=0.
- `test_lens5_doppelganger_chunked_finds_near_identical_pair` — synthetic 10-token vocab with one known-identical pair; Lens 5 surfaces it above threshold.
- `test_lens5_respects_threshold` — pair at similarity 0.94 excluded when threshold=0.95.
- `test_lens6_bridge_score_is_one_when_equidistant` — synthetic token equidistant from two cluster centroids; bridge score ≈ 1.
- `test_lens6_bridge_score_is_zero_when_single_cluster` — token coincident with a cluster centroid; bridge score ≈ 0.

### Framework test

- `test_run_atlas_produces_schema_compliant_json` — synthetic config with tiny real SumerianLookup inputs; verify output JSON schema.

## Performance

Target: <10 min total on the reference laptop.

| Stage | Expected time |
|---|---|
| GloVe load | ~1 min |
| Lens 1 | <1s |
| Lens 2 | ~30s (English Gemma matmul batched) |
| Lens 3 | ~60-90s (chunked 35k² matrix) |
| Lens 4 | ~2-3 min (dual-space) |
| Lens 5 | ~60-90s (threshold-filtered chunks) |
| Lens 6 | ~30s (k-means on 35k × 768) |

## Reproducibility

- Every numeric entry in `anomaly_atlas.json` is a deterministic function of committed inputs + seed.
- Input SHA-256 stamping in JSON so cross-run diffs show whether upstream data changed.
- Atlas markdown auto-regenerates from JSON; prose framing in markdowns is stable across runs.

## Operational notes

**Success path:** Developer runs `python scripts/analysis/sumerian_anomaly_atlas.py`. Script loads artifacts (~1 min), runs 6 lenses (~6-8 min), writes JSON + 7 markdowns (~1s). Exits 0.

**Failure paths:**
- If runtime exceeds 15 min, investigate — likely the chunked cosine computation is tuning suboptimally on this hardware.
- If Lens 5 returns zero pairs above threshold, lower threshold to 0.90 and re-run as a sanity check (implied finding: the alignment has no near-duplicates at the 0.95 mark).
- If Lens 6 bridge scores are all near 0 OR all near 1, k-means k is poorly matched to the vocab; try k=20 or k=80.

## Portability note (Egyptian / comparative repo)

When the comparative repo (or a Gemma-tized Heiroglyphy) needs an atlas:

```python
from cuneiformy.scripts.analysis.anomaly_framework import AnomalyConfig, run_atlas

config = AnomalyConfig(
    civilization_name="egyptian",
    aligned_gemma_path=Path("heiroglyphy/final_output/egyptian_aligned_gemma_vectors.npz"),
    aligned_glove_path=Path("heiroglyphy/final_output/egyptian_aligned_vectors.npz"),
    ...
    junk_target_glosses=frozenset({"x", "n", "det", ...}),  # Egyptian-specific
    output_atlas_json=Path("docs/egyptian_anomaly_atlas.json"),
    ...
)
run_atlas(config)
```

`anomaly_lenses.py` is consumed as-is. No Sumerian-specific logic leaks into either module. User confirmed: when Heiroglyphy is Gemma-tized in the comparative repo, both `aligned_glove_path` and `aligned_gemma_path` will be populated, enabling Lens 4 for Egyptian immediately.

## Follow-up work (out of this spec)

- If atlas rankings surface 10+ interpretable anomalies per lens, promote to a follow-up document ("Anomalies and Outliers in Sumerian Semantic Space") with ~5,000 words of hedged interpretation. That's a separate brainstorm.
- Anchor semantic-quality pass — Lens 1 unfiltered and Lens 4 will likely surface the `sirara→c`, `er3→erra` noise patterns. Remediating them is a separate workstream.
- Egyptian atlas via the comparative repo — sits on top of the Gemma-tized Heiroglyphy workstream, not included here.
