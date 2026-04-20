"""
Sumerian-specific orchestrator for the anomaly atlas framework.

Thin wrapper that builds an AnomalyConfig pointing at Cuneiformy's artifacts
and calls run_atlas. When a future comparative repo or Gemma-tized Heiroglyphy
needs an atlas, it writes its own sibling orchestrator — this file is not
reused.

Run from repo root:
    python scripts/analysis/sumerian_anomaly_atlas.py

See: docs/superpowers/specs/2026-04-20-anomaly-atlas-design.md
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.analysis.anomaly_framework import AnomalyConfig, run_atlas


SUMERIAN_JUNK_ENGLISH_GLOSSES = frozenset({
    "x", "xx", "n", "c", "cf", "unmng", "0", "00", "1", "e", "i", "u", "s",
})


def main() -> int:
    ROOT = _ROOT
    config = AnomalyConfig(
        civilization_name="sumerian",
        aligned_gemma_path=ROOT / "final_output" / "sumerian_aligned_gemma_vectors.npz",
        aligned_glove_path=ROOT / "final_output" / "sumerian_aligned_vectors.npz",
        source_vocab_path=ROOT / "final_output" / "sumerian_aligned_vocab.pkl",
        target_gemma_vocab_path=ROOT / "models" / "english_gemma_whitened_768d.npz",
        target_glove_vocab_path=None,  # GloVe path not needed by Lenses 1-2 in Gemma-only mode
        anchors_path=ROOT / "data" / "processed" / "english_anchors.json",
        corpus_frequency_path=ROOT / "data" / "processed" / "cleaned_corpus.txt",
        junk_target_glosses=SUMERIAN_JUNK_ENGLISH_GLOSSES,
        min_anchor_confidence=0.5,
        min_token_length=2,
        output_atlas_json=ROOT / "docs" / "anomaly_atlas.json",
        output_markdown_dir=ROOT / "docs" / "anomalies",
        output_figures_dir=None,
        seed=42,
        k_clusters=40,
        top_n_per_lens=50,
        doppelganger_threshold=0.95,
        isolation_k=10,
    )
    run_atlas(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
