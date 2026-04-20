"""Consistency tests for anomaly atlas findings document."""
import re
import tempfile
from pathlib import Path

import pytest


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


# --- Numeric-claim tests ----------------------------------------------------


def test_numeric_claim_recognizer_catches_cosines(tmp_path):
    from scripts.docs.consistency import extract_numeric_claims

    md = """
# Test

The cosine similarity of `rin2 -> lord` is -0.088.
Jaccard distance is 1.000.
Bridge score: 0.995.
A bare paragraph with no numbers should not match.
"""
    path = _write(tmp_path, "t.md", md)
    claims = extract_numeric_claims(path)
    assert len(claims) >= 3
    values = [c["value"] for c in claims]
    assert any(abs(v - (-0.088)) < 1e-3 for v in values)
    assert any(abs(v - 1.000) < 1e-3 for v in values)
    assert any(abs(v - 0.995) < 1e-3 for v in values)


def test_numeric_claim_ignores_years_and_integers(tmp_path):
    from scripts.docs.consistency import extract_numeric_claims

    md = """
Published in 1972. Referenced 35,508 tokens.
But the cosine -0.088 is a real claim.
"""
    path = _write(tmp_path, "t.md", md)
    claims = extract_numeric_claims(path)
    # The extractor should focus on decimal numbers in the atlas's value range
    # (typically cosine similarities, Jaccard distances, bridge scores: roughly
    # in [-1.0, 1.0]). Years and large integers are NOT claims.
    values = [c["value"] for c in claims]
    assert any(abs(v - (-0.088)) < 1e-3 for v in values)
    assert not any(abs(v - 1972) < 0.1 for v in values)


def test_numeric_claims_have_json_path_when_atlas_present(tmp_path):
    """Sanity check: when the real atlas is present, verify sample claims trace."""
    import json
    from scripts.docs.consistency import extract_numeric_claims, find_claim_in_atlas

    # Write a tiny atlas JSON
    atlas = {
        "lens1_english_displacement": {
            "rows_filtered": [
                {"sumerian": "rin2", "english": "lord", "cosine_similarity": -0.088},
            ],
        },
    }
    atlas_path = _write(tmp_path, "atlas.json", json.dumps(atlas))

    md_path = _write(tmp_path, "doc.md", "rin2 -> lord has cosine -0.088.")
    claims = extract_numeric_claims(md_path)
    matched = [c for c in claims if find_claim_in_atlas(c, atlas_path)]
    assert len(matched) >= 1


# --- Cuneiform-provenance tests --------------------------------------------


def test_extract_cuneiform_codepoints(tmp_path):
    from scripts.docs.consistency import extract_cuneiform_codepoints

    md = "The sign 𒂗 is EN. Another sign 𒀭 is AN. Regular text has no cuneiform."
    path = _write(tmp_path, "t.md", md)
    codepoints = extract_cuneiform_codepoints(path)
    assert 0x1202D in codepoints  # 𒀭
    assert 0x12097 in codepoints  # 𒂗


def test_every_cuneiform_has_provenance(tmp_path):
    from scripts.docs.consistency import (
        extract_cuneiform_codepoints, check_provenance_coverage,
    )

    md = """
# Document

The sign 𒂗 is EN. See appendix.

## Appendix: cuneiform sign provenance

- U+12097 𒂗 — ePSD2 sign EN, "lord" (source: ORACC)
"""
    path = _write(tmp_path, "complete.md", md)
    codepoints = extract_cuneiform_codepoints(path)
    result = check_provenance_coverage(path, codepoints)
    assert result["covered"] == codepoints
    assert result["missing"] == set()


def test_missing_provenance_flagged(tmp_path):
    from scripts.docs.consistency import (
        extract_cuneiform_codepoints, check_provenance_coverage,
    )

    md = """
# Document

Two signs: 𒂗 and 𒀭. Only one has provenance.

## Appendix: cuneiform sign provenance

- U+12097 𒂗 — ePSD2 sign EN
"""
    path = _write(tmp_path, "partial.md", md)
    codepoints = extract_cuneiform_codepoints(path)
    result = check_provenance_coverage(path, codepoints)
    # 0x1202D 𒀭 is missing from the appendix.
    assert 0x1202D in result["missing"]
