"""
Anomaly-atlas framework: AnomalyConfig + run_atlas orchestrator + markdown renderer.

Civilization-agnostic. Consumed by sumerian_anomaly_atlas.py for Sumerian;
future Egyptian / comparative-repo orchestrators instantiate their own
AnomalyConfig and call run_atlas.

See: docs/superpowers/specs/2026-04-20-anomaly-atlas-design.md
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AnomalyConfig:
    civilization_name: str
    aligned_gemma_path: Path
    aligned_glove_path: Path | None
    source_vocab_path: Path
    target_gemma_vocab_path: Path
    target_glove_vocab_path: Path | None
    anchors_path: Path
    corpus_frequency_path: Path
    junk_target_glosses: frozenset[str]
    min_anchor_confidence: float
    min_token_length: int
    output_atlas_json: Path
    output_markdown_dir: Path
    output_figures_dir: Path | None
    seed: int = 42
    k_clusters: int = 40
    top_n_per_lens: int = 50
    doppelganger_threshold: float = 0.95
    isolation_k: int = 10


# run_atlas lands in a later task.


def _render_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    """Render a list of row-dicts as a markdown table. columns=[(display, key)]."""
    if not rows:
        return "_(no rows)_\n"
    header = "| " + " | ".join(disp for disp, _ in columns) + " |"
    sep = "|" + "|".join(["---"] * len(columns)) + "|"
    lines = [header, sep]
    for row in rows:
        vals = []
        for _, key in columns:
            v = row.get(key, "")
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def render_summary_markdown(atlas: dict) -> str:
    lines = []
    summary = atlas["summary"]
    lines.append(f"# Anomaly Atlas — {atlas['civilization']} ({atlas['atlas_date']})")
    lines.append("")
    lines.append(f"- **Total aligned tokens:** {summary['total_aligned_tokens']:,}")
    lines.append(f"- **Anchor tokens in vocab:** {summary['anchor_tokens_in_vocab']:,}")
    lines.append(f"- **Non-anchor tokens in vocab:** {summary['non_anchor_tokens_in_vocab']:,}")
    lines.append("")
    lines.append("## Top-1 per lens")
    lines.append("")
    for lens_num in range(1, 7):
        key = {
            1: "lens1_english_displacement",
            2: "lens2_no_counterpart",
            3: "lens3_isolation",
            4: "lens4_cross_space_divergence",
            5: "lens5_doppelgangers",
            6: "lens6_structural_bridges",
        }[lens_num]
        top1 = summary.get("top1_per_lens", {}).get(key, "n/a")
        lines.append(f"- **Lens {lens_num} ({key}):** {top1}")
    lines.append("")
    lines.append("## Lens details")
    lines.append("")
    lens_filenames = {
        1: "lens1_english_displacement.md",
        2: "lens2_no_counterpart.md",
        3: "lens3_isolation.md",
        4: "lens4_cross_space_divergence.md",
        5: "lens5_doppelgangers.md",
        6: "lens6_structural_bridges.md",
    }
    for lens_num, fname in lens_filenames.items():
        lines.append(f"- [Lens {lens_num}]({fname})")
    lines.append("")
    return "\n".join(lines)


def render_lens1_markdown(atlas: dict) -> str:
    section = atlas["lens1_english_displacement"]
    lines = [
        f"# Lens 1: English displacement",
        "",
        "Anchor pairs ranked by cosine similarity between the source-language "
        "token's aligned (projected) vector and the English gloss's native target vector. "
        "Low cosine = translation that geometrically misses.",
        "",
        "## Unfiltered (includes anchor-quality noise)",
        "",
        _render_table(
            section["rows_unfiltered"],
            [("Source", "sumerian"), ("English", "english"),
             ("Cos", "cosine_similarity"), ("Conf", "anchor_confidence"),
             ("Src", "source")],
        ),
        "",
        "## Filtered (anchor-quality rules applied)",
        "",
        f"_Rules: {', '.join(section.get('filter_rules_applied', []))}_",
        "",
        _render_table(
            section["rows_filtered"],
            [("Source", "sumerian"), ("English", "english"),
             ("Cos", "cosine_similarity"), ("Conf", "anchor_confidence"),
             ("Src", "source")],
        ),
        "",
    ]
    return "\n".join(lines)


def render_lens2_markdown(atlas: dict) -> str:
    section = atlas["lens2_no_counterpart"]
    lines = [
        "# Lens 2: No counterpart",
        "",
        "Non-anchor source tokens ranked by `corpus_frequency × (1 − top1_target_cosine)`. "
        "High score = appears often in the corpus AND no English word lands close.",
        "",
        _render_table(
            section["rows"],
            [("Source", "sumerian"), ("Freq", "corpus_frequency"),
             ("Top-1 English", "top1_english"),
             ("Cos", "top1_cosine"), ("Score", "score")],
        ),
    ]
    return "\n".join(lines)


def render_lens3_markdown(atlas: dict) -> str:
    section = atlas["lens3_isolation"]
    lines = [
        "# Lens 3: Isolation in source space",
        "",
        "Source tokens ranked by cosine distance to their k-th nearest neighbor. "
        "Large distance = isolated. No alignment needed; pure within-source geometry.",
        "",
        _render_table(
            section["rows"],
            [("Source", "sumerian"), ("D(kth)", "distance_to_kth_neighbor")],
        ),
        "",
        "## Isolation histogram",
        "",
        "_Bin counts (cosine-distance bins across all tokens):_",
        "",
        _render_histogram_line(section["histogram"]),
        "",
    ]
    return "\n".join(lines)


def render_lens4_markdown(atlas: dict) -> str:
    section = atlas["lens4_cross_space_divergence"]
    lines = [
        "# Lens 4: Cross-space divergence",
        "",
        "Source tokens ranked by Jaccard distance between their top-K neighbors "
        "in the Gemma and GloVe aligned spaces. High divergence = one space sees "
        "a facet the other misses (or alignment noise on one side).",
        "",
        "## Unfiltered",
        "",
        _render_table(
            section["rows_unfiltered"],
            [("Source", "sumerian"), ("Jaccard-D", "jaccard_distance")],
        ),
        "",
        "## Anchor-only",
        "",
        _render_table(
            section["rows_anchor_only"],
            [("Source", "sumerian"), ("Jaccard-D", "jaccard_distance")],
        ),
    ]
    return "\n".join(lines)


def render_lens5_markdown(atlas: dict) -> str:
    section = atlas["lens5_doppelgangers"]
    lines = [
        "# Lens 5: Doppelgangers",
        "",
        "Source-token pairs with cosine similarity ≥ threshold. "
        "Near-identical embeddings may indicate morphological variants, "
        "scribal variants, or genuine near-synonyms.",
        "",
        _render_table(
            section["rows"],
            [("Source A", "sumerian_a"), ("Source B", "sumerian_b"),
             ("Cos", "cosine_similarity")],
        ),
        "",
        "## Similarity histogram (≥ 0.85)",
        "",
        _render_histogram_line(section["histogram"]),
        "",
    ]
    return "\n".join(lines)


def render_lens6_markdown(atlas: dict) -> str:
    section = atlas["lens6_structural_bridges"]
    lines = [
        "# Lens 6: Structural bridges",
        "",
        f"Source tokens ranked by bridge score (k-means k={section.get('k_clusters', 'N/A')}, seed={atlas['source_artifacts'].get('seed', 42)}). "
        "Higher score = more equidistant between two clusters = candidate conceptual bridge.",
        "",
        _render_table(
            section["rows"],
            [("Source", "sumerian"), ("Bridge", "bridge_score"),
             ("Cluster A", "nearest_cluster"), ("Cluster B", "second_nearest_cluster")],
        ),
    ]
    return "\n".join(lines)


def _render_histogram_line(hist: dict) -> str:
    """Compact ASCII histogram: bin_start→count."""
    edges = hist.get("bin_edges", [])
    counts = hist.get("counts", [])
    lines = []
    for i, c in enumerate(counts):
        lo = edges[i] if i < len(edges) else 0
        hi = edges[i + 1] if i + 1 < len(edges) else 0
        bar = "█" * min(40, int(c / max(counts) * 40)) if max(counts) > 0 else ""
        lines.append(f"- {lo:.3f}–{hi:.3f}: {c} {bar}")
    return "\n".join(lines)
