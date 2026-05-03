"""Consistency helpers for the anomaly findings document.

Two pure-function utilities used by test_anomaly_findings_consistency.py:
- extract_numeric_claims: scan markdown for likely atlas-number claims
- extract_cuneiform_codepoints: scan markdown for cuneiform Unicode codepoints
- find_claim_in_atlas: cross-reference a numeric claim against the atlas JSON
- check_provenance_coverage: verify each cuneiform codepoint is mentioned in the
  Appendix: cuneiform sign provenance section
"""
from __future__ import annotations

import json
import re
from pathlib import Path


# Cuneiform Unicode blocks:
#   U+12000–U+123FF  Sumero-Akkadian Cuneiform
#   U+12400–U+1247F  Cuneiform Numbers and Punctuation (not used here; see spec)
CUNEIFORM_RANGE = (0x12000, 0x123FF)


def extract_numeric_claims(md_path: Path) -> list[dict]:
    """Extract numeric claims from markdown prose.

    Targets decimals in [-1.0, 1.0] — the typical range of cosine similarities,
    Jaccard distances, and bridge scores in our atlas. Years and large integers
    are excluded.

    Returns a list of {value, context, line_number} dicts.
    """
    text = Path(md_path).read_text(encoding="utf-8")
    pattern = re.compile(r"-?\d+\.\d{1,4}")
    claims = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for match in pattern.finditer(line):
            try:
                value = float(match.group())
            except ValueError:
                continue
            if -1.0 <= value <= 1.0:
                claims.append({
                    "value": value,
                    "context": line.strip()[:120],
                    "line_number": line_no,
                })
    return claims


def extract_cuneiform_codepoints(md_path: Path) -> set[int]:
    """Find every Unicode cuneiform codepoint in the markdown file."""
    text = Path(md_path).read_text(encoding="utf-8")
    lo, hi = CUNEIFORM_RANGE
    return {ord(c) for c in text if lo <= ord(c) <= hi}


def find_claim_in_atlas(claim: dict, atlas_path: Path, tolerance: float = 1e-3) -> bool:
    """Return True if any numeric value in the atlas JSON matches this claim
    within `tolerance`. Recursively walks the atlas dict/list structure."""
    with open(atlas_path) as f:
        atlas = json.load(f)
    target = claim["value"]

    def _walk(obj) -> bool:
        if isinstance(obj, (int, float)):
            try:
                return abs(float(obj) - target) <= tolerance
            except (ValueError, TypeError):
                return False
        if isinstance(obj, dict):
            return any(_walk(v) for v in obj.values())
        if isinstance(obj, list):
            return any(_walk(v) for v in obj)
        return False

    return _walk(atlas)


def check_provenance_coverage(md_path: Path, codepoints: set[int]) -> dict:
    """Verify every cuneiform codepoint is mentioned in the appendix.

    The appendix is identified by the heading "Appendix: cuneiform sign
    provenance" (case-insensitive). Each codepoint entry is expected to include
    "U+" followed by the 5-digit uppercase hex representation.

    Returns {'covered': set[int], 'missing': set[int]}.
    """
    text = Path(md_path).read_text(encoding="utf-8")
    # Split on the appendix heading
    parts = re.split(r"(?i)##\s*appendix.*?provenance.*?\n", text, maxsplit=1)
    appendix_text = parts[1] if len(parts) > 1 else ""

    covered: set[int] = set()
    missing: set[int] = set()
    for cp in codepoints:
        marker = f"U+{cp:05X}"
        if marker in appendix_text or chr(cp) in appendix_text:
            covered.add(cp)
        else:
            missing.add(cp)
    return {"covered": covered, "missing": missing}
