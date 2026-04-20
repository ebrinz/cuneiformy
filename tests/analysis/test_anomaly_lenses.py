"""Unit tests for the civilization-agnostic anomaly atlas framework + lenses."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


def test_anomaly_config_is_frozen():
    from scripts.analysis.anomaly_framework import AnomalyConfig

    config = AnomalyConfig(
        civilization_name="test",
        aligned_gemma_path=Path("/tmp/g.npz"),
        aligned_glove_path=None,
        source_vocab_path=Path("/tmp/vocab.pkl"),
        target_gemma_vocab_path=Path("/tmp/egm.npz"),
        target_glove_vocab_path=None,
        anchors_path=Path("/tmp/a.json"),
        corpus_frequency_path=Path("/tmp/corp.txt"),
        junk_target_glosses=frozenset({"x", "n"}),
        min_anchor_confidence=0.5,
        min_token_length=2,
        output_atlas_json=Path("/tmp/out.json"),
        output_markdown_dir=Path("/tmp/md"),
        output_figures_dir=None,
    )
    with pytest.raises((AttributeError, Exception)):
        config.civilization_name = "hacked"
