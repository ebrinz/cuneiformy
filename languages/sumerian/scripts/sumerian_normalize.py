"""
Canonical Sumerian token normalization.

Single source of truth for mapping ORACC citation forms and inflected surface
forms to the common ATF-based token form produced by
`scripts/05_clean_and_tokenize.py`.

Used by `scripts/06_extract_anchors.py` (anchor side) and
`scripts/coverage_diagnostic.py` (audit/diagnostic side). Keeping this function
in one place prevents normalization drift between anchors and corpus.

See: docs/superpowers/specs/2026-04-19-workstream-2b-normalization-fix-design.md
"""
from __future__ import annotations

import re

_SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

_ORACC_TO_ATF = {
    "š": "sz", "Š": "SZ",
    "ŋ": "j",  "Ŋ": "J",
    "ḫ": "h",  "Ḫ": "H",
    "ṣ": "s",  "Ṣ": "S",
    "ṭ": "t",  "Ṭ": "T",
    "ʾ": "",
}

_BRACE_RE = re.compile(r"\{([^}]*)\}")


def normalize_sumerian_token(raw) -> str:
    """Canonical normalization for a single Sumerian token.

    Applies (in order):
      1. Unicode subscript digits -> ASCII digits
      2. Strip determinative braces {X} keeping content
      3. ORACC Sumerian unicode letters -> ATF (š -> sz, etc.)
      4. Drop hyphens (produces fully-joined compound form)
      5. Lowercase + strip whitespace

    Safe on None and empty input (returns "").
    Idempotent: normalize(normalize(x)) == normalize(x).
    """
    if raw is None:
        return ""
    s = str(raw)
    s = s.translate(_SUBSCRIPT_MAP)
    s = _BRACE_RE.sub(r"\1", s)
    for old, new in _ORACC_TO_ATF.items():
        s = s.replace(old, new)
    s = s.replace("-", "")
    return s.lower().strip()
