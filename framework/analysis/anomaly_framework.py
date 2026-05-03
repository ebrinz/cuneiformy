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

import datetime as _dt
import hashlib
import json as _json_mod
import pickle as _pkl_mod
from collections import Counter

import numpy as _np_mod

from framework.analysis.anomaly_lenses import (
    lens1_english_displacement, lens2_no_counterpart, lens3_isolation,
    lens4_cross_space_divergence, lens5_doppelgangers, lens6_structural_bridges,
)


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_matrix(X):
    norms = _np_mod.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def _load_aligned_npz(path):
    data = _np_mod.load(path, allow_pickle=True)
    if "vectors" in data.files:
        v = data["vectors"].astype(_np_mod.float32)
    else:
        raise ValueError(f"expected 'vectors' array in {path}")
    return _normalize_matrix(v)


def _load_target_gemma_npz(path):
    data = _np_mod.load(path, allow_pickle=True)
    vocab = [str(w) for w in data["vocab"]]
    vectors = _normalize_matrix(data["vectors"].astype(_np_mod.float32))
    return vocab, vectors


def _load_vocab_pkl(path):
    with open(path, "rb") as f:
        return list(_pkl_mod.load(f))


def _load_corpus_frequency(path):
    freq = Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            for token in line.strip().split():
                freq[token] += 1
    return dict(freq)


def _load_anchors(path):
    with open(path, encoding="utf-8") as f:
        return _json_mod.load(f)


def run_atlas(config) -> dict:
    """Orchestrate the 6 lenses, render JSON + markdown, return the summary dict.

    Lens 4 is skipped with a note if aligned_glove_path is None.
    """
    print(f"[atlas] Loading artifacts...")
    aligned_gemma = _load_aligned_npz(config.aligned_gemma_path)
    source_vocab = _load_vocab_pkl(config.source_vocab_path)
    target_gemma_vocab, target_gemma_vectors = _load_target_gemma_npz(config.target_gemma_vocab_path)
    target_gemma_vocab_map = {w.lower(): i for i, w in enumerate(target_gemma_vocab)}
    anchors = _load_anchors(config.anchors_path)
    corpus_freq = _load_corpus_frequency(config.corpus_frequency_path)

    anchor_source_tokens = frozenset(
        a["sumerian"] for a in anchors if a.get("sumerian")
    )

    aligned_glove = None
    if config.aligned_glove_path:
        aligned_glove = _load_aligned_npz(config.aligned_glove_path)

    # Lens 1
    print(f"[atlas] Lens 1 (english displacement)...")
    l1 = lens1_english_displacement(
        aligned_gemma, source_vocab, target_gemma_vectors, target_gemma_vocab_map,
        anchors, config.top_n_per_lens, config.junk_target_glosses,
        config.min_token_length, config.min_anchor_confidence,
    )

    # Lens 2
    print(f"[atlas] Lens 2 (no counterpart)...")
    l2 = lens2_no_counterpart(
        aligned_gemma, source_vocab, anchor_source_tokens,
        target_gemma_vectors, target_gemma_vocab, corpus_freq,
        config.top_n_per_lens,
    )

    # Lens 3
    print(f"[atlas] Lens 3 (isolation)...")
    l3 = lens3_isolation(
        aligned_gemma, source_vocab, config.isolation_k, config.top_n_per_lens,
    )

    # Lens 4
    if aligned_glove is not None:
        print(f"[atlas] Lens 4 (cross-space divergence)...")
        l4 = lens4_cross_space_divergence(
            aligned_gemma, aligned_glove, source_vocab, anchor_source_tokens,
            config.top_n_per_lens, neighbors_k=10,
        )
    else:
        print(f"[atlas] Lens 4 skipped (no aligned_glove_path)")
        l4 = {"rows_unfiltered": [], "rows_anchor_only": [],
              "note": "skipped (requires dual-target alignment)"}

    # Lens 5
    print(f"[atlas] Lens 5 (doppelgangers)...")
    l5 = lens5_doppelgangers(
        aligned_gemma, source_vocab, anchor_source_tokens,
        config.doppelganger_threshold, config.top_n_per_lens,
    )

    # Lens 6
    print(f"[atlas] Lens 6 (structural bridges)...")
    l6 = lens6_structural_bridges(
        aligned_gemma, source_vocab, config.k_clusters, config.top_n_per_lens,
        seed=config.seed,
    )

    total_tokens = len(source_vocab)
    anchor_count_in_vocab = sum(1 for t in source_vocab if t in anchor_source_tokens)
    non_anchor_count = total_tokens - anchor_count_in_vocab

    def _top1_lens1():
        rows = l1.get("rows_unfiltered", [])
        if not rows:
            return "n/a"
        r = rows[0]
        return f"{r['sumerian']} -> {r['english']} (cos={r['cosine_similarity']:.4f})"

    def _top1_lens2():
        rows = l2.get("rows", [])
        if not rows:
            return "n/a"
        r = rows[0]
        return f"{r['sumerian']} (freq={r['corpus_frequency']}, top1_cos={r['top1_cosine']:.4f})"

    def _top1_lens3():
        rows = l3.get("rows", [])
        if not rows:
            return "n/a"
        r = rows[0]
        return f"{r['sumerian']} (d_k={r['distance_to_kth_neighbor']:.4f})"

    def _top1_lens4():
        rows = l4.get("rows_unfiltered", [])
        if not rows:
            return "n/a (skipped)" if "note" in l4 else "n/a"
        r = rows[0]
        return f"{r['sumerian']} (jaccard={r['jaccard_distance']:.4f})"

    def _top1_lens5():
        rows = l5.get("rows", [])
        if not rows:
            return "n/a"
        r = rows[0]
        return f"{r['sumerian_a']} == {r['sumerian_b']} (cos={r['cosine_similarity']:.4f})"

    def _top1_lens6():
        rows = l6.get("rows", [])
        if not rows:
            return "n/a"
        r = rows[0]
        return (f"{r['sumerian']} (bridge={r['bridge_score']:.4f}, "
                f"clusters {r['nearest_cluster']}/{r['second_nearest_cluster']})")

    atlas = {
        "atlas_schema_version": 1,
        "atlas_date": _dt.date.today().isoformat(),
        "civilization": config.civilization_name,
        "source_artifacts": {
            "aligned_gemma_path": str(config.aligned_gemma_path),
            "aligned_glove_path": str(config.aligned_glove_path) if config.aligned_glove_path else None,
            "source_vocab_path": str(config.source_vocab_path),
            "anchors_path": str(config.anchors_path),
            "anchors_sha256": _sha256(config.anchors_path),
            "corpus_frequency_path": str(config.corpus_frequency_path),
            "corpus_frequency_sha256": _sha256(config.corpus_frequency_path),
            "seed": config.seed,
            "k_clusters": config.k_clusters,
            "top_n_per_lens": config.top_n_per_lens,
            "doppelganger_threshold": config.doppelganger_threshold,
            "isolation_k": config.isolation_k,
        },
        "summary": {
            "total_aligned_tokens": total_tokens,
            "anchor_tokens_in_vocab": anchor_count_in_vocab,
            "non_anchor_tokens_in_vocab": non_anchor_count,
            "top1_per_lens": {
                "lens1_english_displacement": _top1_lens1(),
                "lens2_no_counterpart": _top1_lens2(),
                "lens3_isolation": _top1_lens3(),
                "lens4_cross_space_divergence": _top1_lens4(),
                "lens5_doppelgangers": _top1_lens5(),
                "lens6_structural_bridges": _top1_lens6(),
            },
        },
        "lens1_english_displacement": l1,
        "lens2_no_counterpart": l2,
        "lens3_isolation": l3,
        "lens4_cross_space_divergence": l4,
        "lens5_doppelgangers": l5,
        "lens6_structural_bridges": l6,
    }

    # Write JSON (sort keys for determinism)
    config.output_atlas_json.parent.mkdir(parents=True, exist_ok=True)
    with open(config.output_atlas_json, "w") as f:
        _json_mod.dump(atlas, f, indent=2, sort_keys=True)
        f.write("\n")

    # Render and write markdown files
    config.output_markdown_dir.mkdir(parents=True, exist_ok=True)
    (config.output_markdown_dir / "atlas_summary.md").write_text(render_summary_markdown(atlas))
    (config.output_markdown_dir / "lens1_english_displacement.md").write_text(render_lens1_markdown(atlas))
    (config.output_markdown_dir / "lens2_no_counterpart.md").write_text(render_lens2_markdown(atlas))
    (config.output_markdown_dir / "lens3_isolation.md").write_text(render_lens3_markdown(atlas))
    (config.output_markdown_dir / "lens4_cross_space_divergence.md").write_text(render_lens4_markdown(atlas))
    (config.output_markdown_dir / "lens5_doppelgangers.md").write_text(render_lens5_markdown(atlas))
    (config.output_markdown_dir / "lens6_structural_bridges.md").write_text(render_lens6_markdown(atlas))

    print(f"[atlas] Wrote {config.output_atlas_json}")
    print(f"[atlas] Wrote {config.output_markdown_dir}/*.md")
    return atlas["summary"]
