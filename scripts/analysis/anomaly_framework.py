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


# run_atlas and markdown renderer land in a later task.
