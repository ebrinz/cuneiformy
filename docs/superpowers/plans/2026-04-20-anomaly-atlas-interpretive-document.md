# Anomaly Atlas Interpretive Document Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `docs/anomaly_atlas_findings.md` (~9,500-word thematic interpretation of the anomaly atlas) plus its PDF rendering via pandoc + xelatex with embedded Noto Sans Cuneiform font, plus a reusable build script and pandoc template for future research documents.

**Architecture:** Four-phase build — (1) infrastructure: font + template + build script + smoke test; (2) consistency tests against synthetic markdown; (3) the prose document itself, written with numeric claims traced to `docs/anomaly_atlas.json` and cuneiform signs sourced from ePSD2; (4) real-data PDF render + commit + journal. Standalone document — does not re-engage with the cosmogony document's thesis.

**Tech Stack:** Markdown, pandoc ≥ 3.0, xelatex (MacTeX / TeX Live), Noto Sans Cuneiform font (OFL-licensed), pytest. No new Python dependencies.

**Reference spec:** `docs/superpowers/specs/2026-04-20-anomaly-atlas-interpretive-document-design.md`

---

## Before You Begin

- Current branch: `master`. Cut a fresh feature branch:
  ```bash
  cd /Users/crashy/Development/cuneiformy
  git checkout -b feat/anomaly-findings-doc
  ```
  All commits land on `feat/anomaly-findings-doc`. Merge via `superpowers:finishing-a-development-branch` after Task 4.

- Verify system dependencies that will be needed in Task 1 + Task 4:
  ```bash
  command -v pandoc && pandoc --version | head -1
  command -v xelatex && xelatex --version | head -1
  ```
  Both must be present; if either is missing, install before Task 1. If you cannot install xelatex in this environment, report BLOCKED at Task 1.

- Verify the anomaly atlas artifact is present (the doc cites data from this file):
  ```bash
  ls -la docs/anomaly_atlas.json
  python3 -c "import json; d = json.load(open('docs/anomaly_atlas.json')); print('civilization:', d['civilization'])"
  ```

---

## File Structure

**New files:**
- `docs/fonts/NotoSansCuneiform-Regular.ttf` — OFL-licensed font, ~1.5MB (committed, force-add if necessary).
- `docs/templates/cuneiformy-pandoc.tex` — pandoc LaTeX template with cuneiform auto-switching.
- `scripts/docs/render_anomaly_pdf.sh` — build script with preflight checks.
- `docs/anomaly_atlas_findings.md` — the ~9,500-word prose document (13 sections).
- `docs/anomaly_atlas_findings.pdf` — rendered PDF (committed; regenerable from the markdown).
- `tests/test_anomaly_findings_consistency.py` — 2 consistency tests.

**Modified files:**
- `docs/EXPERIMENT_JOURNAL.md` — journal entry on completion.
- `README.md` — add link to the document under Research Progress.

**Untouched:**
- All prior pipeline scripts, analysis modules, tests, cosmogony document, atlas JSON.

---

## Task 1: Infrastructure — font, template, build script, smoke test

**Files:**
- Create: `docs/fonts/NotoSansCuneiform-Regular.ttf`
- Create: `docs/templates/cuneiformy-pandoc.tex`
- Create: `scripts/docs/render_anomaly_pdf.sh`

### Setup note

This task sets up the build pipeline and verifies it works with a small smoke-test markdown file before any real prose is written. Smoke test is NOT committed — it's a temp file to confirm the font + template + script work end-to-end.

- [ ] **Step 1: Download and commit Noto Sans Cuneiform font**

```bash
cd /Users/crashy/Development/cuneiformy
mkdir -p docs/fonts
curl -L -o docs/fonts/NotoSansCuneiform-Regular.ttf \
  "https://raw.githubusercontent.com/notofonts/cuneiform/main/fonts/NotoSansCuneiform/full/ttf/NotoSansCuneiform-Regular.ttf"
```

Verify the download:
```bash
file docs/fonts/NotoSansCuneiform-Regular.ttf
ls -la docs/fonts/NotoSansCuneiform-Regular.ttf
```
Expected: file is ~1.5-2MB, TrueType font type identification. If the curl URL 404s, fall back to:
```bash
curl -L -o /tmp/noto-cuneiform.zip "https://fonts.google.com/download?family=Noto%20Sans%20Cuneiform"
unzip -p /tmp/noto-cuneiform.zip "NotoSansCuneiform/NotoSansCuneiform-Regular.ttf" > docs/fonts/NotoSansCuneiform-Regular.ttf
```

- [ ] **Step 2: Create the pandoc LaTeX template**

Create `docs/templates/cuneiformy-pandoc.tex`:

```latex
\documentclass[$if(fontsize)$$fontsize$,$endif$]{article}

\usepackage{fontspec}
\usepackage{geometry}
\usepackage{hyperref}
\usepackage{longtable,booktabs,array}
\usepackage{microtype}
\usepackage{parskip}
\usepackage{graphicx}
\usepackage[normalem]{ulem}
\usepackage{calc}

$if(geometry)$
\geometry{$geometry$}
$endif$

$if(mainfont)$
\setmainfont{$mainfont$}
$endif$
$if(monofont)$
\setmonofont{$monofont$}
$endif$

% Noto Sans Cuneiform for Unicode cuneiform (U+12000..U+123FF).
% Path is relative to the command's --resource-path resolution.
\newfontfamily{\cuneiformfont}{NotoSansCuneiform-Regular.ttf}[Path=./docs/fonts/]

% XeTeX inter-character class trick: auto-switch to cuneiform font for any
% character in the Sumero-Akkadian Cuneiform Unicode block.
\XeTeXinterchartokenstate=1
\newXeTeXintercharclass\CuneiformClass
\XeTeXcharclass "12000 \CuneiformClass
\XeTeXcharclass "12001 \CuneiformClass
% ... Use a range via a Lua-less approach below.
% Simpler: define the class over the whole block programmatically via \count.
\makeatletter
\@tempcnta=\"12000
\@tempcntb=\"123FF
\loop\ifnum\@tempcnta<\@tempcntb
  \XeTeXcharclass\@tempcnta\CuneiformClass
  \advance\@tempcnta by 1
\repeat
\XeTeXcharclass\@tempcntb\CuneiformClass
\makeatother

\XeTeXinterchartoks 0 \CuneiformClass = {\begingroup\cuneiformfont}
\XeTeXinterchartoks \CuneiformClass 0 = {\endgroup}

\hypersetup{
  colorlinks=true,
  linkcolor=blue!70!black,
  urlcolor=blue!70!black,
}

\title{$title$}
\author{$for(author)$$author$$sep$ \\ $endfor$}
\date{$date$}

\begin{document}

$if(title)$
\maketitle
$endif$

$if(toc)$
\tableofcontents
\newpage
$endif$

$body$

\end{document}
```

This is a minimal pandoc template; pandoc's default template has many more options but for a straightforward document, this is enough. The key lines are the `\newfontfamily{\cuneiformfont}` declaration and the `\XeTeXinterchartoks` block that auto-switches fonts around cuneiform codepoints.

- [ ] **Step 3: Create the build script**

Create `scripts/docs/render_anomaly_pdf.sh`:

```bash
#!/usr/bin/env bash
# Render docs/anomaly_atlas_findings.md to PDF via pandoc + xelatex.
#
# Preflight checks pandoc, xelatex, and the embedded Noto Sans Cuneiform font.
# On success, produces docs/anomaly_atlas_findings.pdf.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# Preflight
fail=0
if ! command -v pandoc >/dev/null 2>&1; then
  echo "ERROR: pandoc not found. Install: brew install pandoc (macOS) or apt install pandoc (Debian)." >&2
  fail=1
fi
if ! command -v xelatex >/dev/null 2>&1; then
  echo "ERROR: xelatex not found. Install MacTeX (macOS: brew install --cask mactex-no-gui) or texlive-xetex (Debian)." >&2
  fail=1
fi
if [ ! -f "docs/fonts/NotoSansCuneiform-Regular.ttf" ]; then
  echo "ERROR: docs/fonts/NotoSansCuneiform-Regular.ttf missing. Run the Task 1 Step 1 curl in the plan." >&2
  fail=1
fi
if [ ! -f "docs/anomaly_atlas_findings.md" ]; then
  echo "ERROR: docs/anomaly_atlas_findings.md missing. Write the prose (Task 3) before rendering the PDF." >&2
  fail=1
fi
if [ "$fail" -ne 0 ]; then
  exit 1
fi

OUT="docs/anomaly_atlas_findings.pdf"

echo "[render] pandoc ..."
pandoc docs/anomaly_atlas_findings.md \
  --pdf-engine=xelatex \
  --template=docs/templates/cuneiformy-pandoc.tex \
  --resource-path=docs/fonts:docs \
  --variable=fontsize:11pt \
  --variable=geometry:margin=1in \
  --variable=mainfont:"Noto Sans" \
  --variable=monofont:"Noto Sans Mono" \
  --number-sections \
  --toc \
  --toc-depth=2 \
  -o "$OUT"

if [ ! -s "$OUT" ]; then
  echo "ERROR: PDF was not produced or is empty." >&2
  exit 1
fi

echo "[render] Wrote $OUT ($(wc -c < "$OUT") bytes)"
```

Make it executable:
```bash
chmod +x scripts/docs/render_anomaly_pdf.sh
```

- [ ] **Step 4: Smoke-test the pipeline with a tiny markdown file**

Create a temp test markdown:
```bash
cat > /tmp/smoke_cuneiform.md <<'MD'
---
title: Cuneiform Render Smoke Test
---

# Smoke test

The sign 𒂗 is EN (lord).

The sign 𒀭 is AN (heaven).

A word in cuneiform: 𒀭𒂗𒆤 (*den-lil2*, the god Enlil).

Plain English text still renders with the main font.
MD
```

Run pandoc directly (don't use the script since it looks for the real markdown file):
```bash
pandoc /tmp/smoke_cuneiform.md \
  --pdf-engine=xelatex \
  --template=docs/templates/cuneiformy-pandoc.tex \
  --resource-path=docs/fonts:docs \
  --variable=fontsize:11pt \
  --variable=geometry:margin=1in \
  --variable=mainfont:"Noto Sans" \
  --variable=monofont:"Noto Sans Mono" \
  -o /tmp/smoke_cuneiform.pdf
```

Expected: /tmp/smoke_cuneiform.pdf produced without LaTeX errors. Open it (macOS: `open /tmp/smoke_cuneiform.pdf`) and visually confirm:
- The three cuneiform signs render as actual cuneiform (not boxes).
- English text uses the main font.
- No blank pages, no LaTeX compilation errors in pandoc stderr.

If the cuneiform renders as boxes, the font isn't being picked up — check:
  - Font file path (relative to `--resource-path`)
  - `\newfontfamily` line in the template
  - xelatex version supports `\XeTeXcharclass`

If everything works, delete the smoke-test files:
```bash
rm /tmp/smoke_cuneiform.md /tmp/smoke_cuneiform.pdf
```

- [ ] **Step 5: Full test suite regression check**

```bash
pytest tests/ --ignore=tests/test_integration.py -q
```
Expected: 161 pass. No new tests yet; this just confirms nothing was broken inadvertently.

- [ ] **Step 6: Commit**

```bash
cd /Users/crashy/Development/cuneiformy
git add docs/fonts/NotoSansCuneiform-Regular.ttf \
        docs/templates/cuneiformy-pandoc.tex \
        scripts/docs/render_anomaly_pdf.sh
git commit -m "feat: add cuneiform PDF rendering pipeline (font + template + script)"
```

Note: the font file is binary. If the repo has git-lfs, use it; otherwise regular git commit is fine (~1.5MB is small enough).

---

## Task 2: Consistency tests (TDD against synthetic markdown)

**Files:**
- Create: `tests/test_anomaly_findings_consistency.py`

### Setup note

Both tests parse a markdown file and check consistency against external sources:
- Numeric claims in prose should cross-reference atlas JSON paths.
- Cuneiform characters used in prose should appear in a provenance appendix.

Tests use synthetic markdown content so they pass before the real document is written. When the real document exists, the tests additionally cover it.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_anomaly_findings_consistency.py`:

```python
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
    assert 0x12017 in codepoints  # 𒀭
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
    # 0x12017 𒀭 is missing from the appendix.
    assert 0x12017 in result["missing"]
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
cd /Users/crashy/Development/cuneiformy
pytest tests/test_anomaly_findings_consistency.py -v
```
Expected: 5 FAIL with `ModuleNotFoundError: No module named 'scripts.docs.consistency'`.

- [ ] **Step 3: Implement the helpers**

Create `scripts/docs/__init__.py` (empty) and `scripts/docs/consistency.py`:

```python
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
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/test_anomaly_findings_consistency.py -v
```
Expected: 5 PASS.

- [ ] **Step 5: Full suite regression**

```bash
pytest tests/ --ignore=tests/test_integration.py -q
```
Expected: 166 PASS (161 + 5 new).

- [ ] **Step 6: Commit**

```bash
cd /Users/crashy/Development/cuneiformy
git add scripts/docs/__init__.py scripts/docs/consistency.py tests/test_anomaly_findings_consistency.py
git commit -m "feat: add anomaly findings consistency helpers + tests"
```

---

## Task 3: Write the document prose

**Files:**
- Create: `docs/anomaly_atlas_findings.md`

### Setup note

This is the big task — ~9,500 words of prose grounded in `docs/anomaly_atlas.json`. Strict discipline: every number traces to a JSON path, every Sumerological claim cites ePSD2 or named secondary, every cuneiform sign gets a provenance entry in §12.

### Data you MUST reference before writing

Before writing any prose, read these files and keep them open:

1. `docs/anomaly_atlas.json` — the canonical atlas data. Every numeric claim in prose comes from here.
2. `docs/anomalies/lens1_english_displacement.md` through `lens6_structural_bridges.md` — pre-formatted tables from the atlas.
3. `docs/superpowers/specs/2026-04-20-anomaly-atlas-interpretive-document-design.md` — the document structure + theme definitions.
4. `docs/EXPERIMENT_JOURNAL.md` — prior Workstream 2a/2b findings you may reference.
5. `results/cosmogony_preflight_2026-04-19.json` — prior cosmogony preflight; you don't cite it but the ETCSL passage count info might be useful cross-reference if specific Sumerian terms overlap.

### Key data points (already captured)

Top-tier findings per the atlas baseline (pinned commit `ff94533`):

| Lens | Top filtered finding | Value |
|---|---|---|
| 1 | `rin2 → lord` | cosine -0.088 |
| 1 | `jizzal → ear` | cosine -0.063 |
| 1 | `en → priest` | cosine -0.037 |
| 1 | `du14 → combat` | cosine -0.033 |
| 2 | `sze3 → father` | freq 52,685, top1_cos 0.291 (filtered — top numeric candidates excluded as corpus artifact) |
| 3 | `had2` | d_10 = 0.633 |
| 3 | `šir` | d_10 = 0.622 |
| 3 | `uttu` | d_10 = 0.622 |
| 3 | `gidri` | d_10 = 0.619 |
| 3 | `ebgal` | d_10 = 0.616 |
| 4 | `asag` | Jaccard 1.000 (anchor-only list) |
| 4 | `amra`, `dubkam`, `dumdam`, `dupsik` | Jaccard 1.000 — several strong candidates |
| 5 | `baandab5be2ec == baandab5be2eš` | cos 0.998 |
| 5 | `ciingaantum2mu == šiingaantum2mu` | cos 0.994 |
| 6 | `ningir2su2kake4` | bridge 1.000, clusters 9/4 |
| 6 | `kas4ke4nesze3` | bridge 1.000, clusters 3/29 |
| 6 | `karzidda` | bridge 1.000, clusters 33/1 |

Pick case studies from these. Favor anchors with high confidence values (≥ 0.9) from atlas rows when selecting Lens 1 cases; high confidence + large displacement is the most interpretive-rich combination.

### Cuneiform signs — planned provenance

Source each via ePSD2 (http://oracc.museum.upenn.edu/epsd2/) where possible; fall back to the sign lists in Jeremy Black, Graham Cunningham, Eleanor Robson & Gábor Zólyomi, *The Literature of Ancient Sumer* (Oxford, 2004) or Borger, *Mesopotamisches Zeichenlexikon*.

Minimal sign list you will likely need — curate provenance for at least these ~20-25 signs:

- `𒂗` (EN, U+12097) — "lord", sign EN
- `𒀭` (AN, U+12017) — "god/heaven", sign AN (determinative for deities)
- `𒄑` (GIŠ, U+12117) — "wood/tree" (determinative for wooden objects, e.g., `gidri`)
- `𒆠` (KI, U+121A0) — "place/earth" (determinative for place names)
- `𒀸` (AŠ, U+12038) — "one" / numeric marker
- `𒄀` (ZI, U+12100) — "life/breath/reading"
- `𒊮` (ŠA, U+122EE) — "heart" or sometimes phonetic marker
- `𒉺𒅗` (for `karzidda`) — kar + zid (true-quay)
- `𒊩` (MUNUS, U+122A9) — "woman" (determinative)
- `𒌓` (UD, U+1230D) — "day/sun"
- Others emerge from the case studies you pick.

### Writing workflow

Write section by section. After each section, re-read it and verify:
- Numeric claims: pick any number you cited; find its path in `docs/anomaly_atlas.json`.
- Sumerological claims: is there an ETCSL text ID or a named secondary cited nearby?
- Cuneiform codepoints: noted in §12 appendix.

Commit after every 2-3 sections so you can incrementally verify progress.

- [ ] **Step 1: Write §0 Abstract + §1 Introduction + §2 Methodology**

Target: ~1,200 words. Cover:
- Abstract (200): what the doc does, what it finds at the highest level, what it is not (standalone, not a revision of the cosmogony).
- Introduction (500): what the atlas is, what "anomaly" means in this context, the six themes at a glance.
- Methodology (500): brief pipeline description, link to the atlas spec, caveats (alignment quality at 52% top-1, anchor-quality issues surfaced via Lens 1 unfiltered, corpus bias toward literary Sumerian).

Cite the pinned atlas commit `ff94533` in §2.

- [ ] **Step 2: Write §3 Theme 1 — Translation failures**

Target: ~1,800 words. Three case studies, each ~500-600 words. Recommended:
1. `en → priest` — the easy case. EN is traditionally translated as "priest" for the high priestess of Inanna, but `en` is also a political title (city-ruler in early Dynastic period) and the name of a god (Enki is `d-en-ki`, "lord earth"). The atlas shows cos -0.037 — the English projection of "priest" lands geometrically away from what `en`'s actual usage geometry suggests. Discuss the conflation of political-religious-divine that English "priest" can't carry.
2. `jizzal → ear` — metonymic loss. `jizzal` appears to mean wisdom/intelligence in Sumerian (body-concept of ear as seat of wisdom). English "ear" reduces this to physical organ. Cos -0.063.
3. `rin2 → lord` — highest confidence anchor with most negative cosine. `rin2` has narrower semantic range than "lord" and the geometry reflects that.

Include cuneiform signs where known: `𒂗` (en), possibly `𒉿` (for `jizzal` if signs known), the sign for rin2.

Heading uses `𒂗` as the theme sign.

- [ ] **Step 3: Write §4 Theme 2 — Specialized cultic vocabulary**

Target: ~1,500 words. Three or four case studies, each ~400 words.

Candidates from Lens 3 top-10 isolation:
- `uttu` — goddess of weaving. A specific deity; isolation because her semantic neighborhood is narrow.
- `gidri` — scepter, sign of office. `𒄑` determinative (wooden object).
- `ebgal` — "great shrine" (possibly a specific location name or a generic sacred architectural term).
- `asag` — from Lens 4 anchor-only (Jaccard 1.000). The Asag-demon in Lugalbanda and Hurrum cycle; famously translated "demon" but the concept is more specific (a cosmic chaos principle).

Discuss the pattern: specialized concepts that resist English reduction because their cultic/mythological function is itself culturally specific.

Heading uses `𒄑` as theme sign.

- [ ] **Step 4: Write §5 Theme 3 — Grammatical bridges**

Target: ~1,500 words. Three case studies, each ~500 words.

From Lens 6 top-10 (bridge = 1.000):
1. `ningir2su2kake4` — Ningirsu + dative case + connective suffix. The deity's name in a grammatical case, which puts it geometrically between the deity-concept cluster and the case-suffix cluster. Discuss how FastText can only see surface strings, so this compound form IS a semantic bridge in the token-level representation.
2. `kas4ke4nesze3` — `kas4` (runner) + `-ke4` (genitive) + `-ne-` (plural suffix) + `-sze3` (directional "toward"). Four morphemes stacked; the token sits between runner-concepts and grammatical-postposition concepts.
3. `karzidda` — `kar` (quay) + `zid` (true/real). A compound noun for a specific place or concept. Bridges between place-vocabulary and epithet-vocabulary.

Discuss the meta-pattern: agglutinative morphology creates bridge tokens in the aligned space.

Heading uses `𒆠` as theme sign.

- [ ] **Step 5: Write §6 Theme 4 — Transliteration shadows**

Target: ~800 words, shorter than the main themes. Single rigorous discussion.

From Lens 5 top-10, note the `c/š` variant pairs: `baandab5be2ec == baandab5be2eš` (cos 0.998), `ciingaantum2mu == šiingaantum2mu` (cos 0.994), `imciingi4gi4 == imšiingi4gi4` (cos 0.992). ORACC uses `š` in its citation forms; ATF (the bulk transliteration convention) uses `sz` or `c` depending on the transliterator. These are the SAME word with different surface strings; the fact that the embedding places them at cos > 0.99 is a sanity check — the character n-gram structure of FastText connects them even without being told they're variants.

Briefly note that `5(n01@f) == 6(n01@f)` doesn't have the same story — it's a numeric-notation pair that the embedding has learned to confuse (not a sanity-check positive but a numeric-artifact issue).

Heading uses `𒊮` as theme sign.

- [ ] **Step 6: Write §7 Theme 5 — The numeric tail**

Target: ~800 words, shorter.

Lens 2 (non-anchor high-value terms) top-10 is entirely numeric forms: `1(disz)`, `1(u)`, `2(disz)`, `5(disz)`, `3(disz)`, `2(u)`, `1(asz@c)`, `4(disz)`, `3(u)`. These appear 22k-115k times in the corpus each. Why: the Sumerian tablet corpus is dominated by administrative and accounting texts that count commodities. `(disz)` = singular count marker, `(u)` = "tens" count marker, `(asz@c)` = another numeric notation.

Discuss the interpretive caveat: these tokens are corpus-artifacts, not findings about Sumerian cognition. They illustrate the gap between "high corpus frequency × low English match" (the Lens 2 score) and "high-value conceptual term." A more nuanced Lens 2 would weight against numeric-form markers, but that's filter engineering — we choose to show the raw signal and discuss.

Heading uses `𒀸` as theme sign.

- [ ] **Step 7: Write §8 Theme 6 — Reading through the floor (meta)**

Target: ~1,000 words.

Meta-essay on how to read the atlas. Central thesis: the top-1 row of each lens is usually NOT the most interesting finding. Top-1 tends to be:
- Lens 1: hyper-confident anchors with mild translation issues (or anchor-quality noise)
- Lens 2: corpus-frequency artifacts
- Lens 3: obscure rare words
- Lens 4: tiny tokens with noisy alignment in both spaces
- Lens 5: transliteration variants (ORACC convention shadows)
- Lens 6: proper names in grammatical cases

The RICH MIDDLE of the rankings (say, ranks 15-35) is where interpretation lives — the tokens where the lens's signal is above noise but not dominated by an obvious artifact. Argue this pattern holds across lenses and suggest that a future document on atlas-driven findings should look there preferentially.

Heading uses `𒄀` as theme sign.

- [ ] **Step 8: Write §9 Synthesis + §10 Reproducibility + §11 References + §12 Appendix**

Target: ~1,000 words + reference list + cuneiform provenance.

§9 Synthesis (~500): What the six themes collectively reveal. The atlas is a better diagnostic than a ranker — it surfaces patterns (translation failures, grammatical bridges, transliteration sanity checks) rather than individual "discoveries." Cite RESEARCH_VISION.md's thesis about the losses/gains of civilizational time and note that the atlas data is consistent with it but doesn't prove it.

§10 Reproducibility (~200): Pinned atlas commit `ff94533`. Instructions for regenerating: `python scripts/analysis/sumerian_anomaly_atlas.py`. PDF regeneration: `bash scripts/docs/render_anomaly_pdf.sh`.

§11 References: ETCSL text IDs actually cited in the document. ePSD2 entries. Jacobsen (1976) *The Treasures of Darkness*. Black et al. (2004) *The Literature of Ancient Sumer*. Kramer (1944) *Sumerian Mythology*. Foxvog *Introduction to Sumerian Grammar*. Borger *Mesopotamisches Zeichenlexikon*.

§12 Appendix — cuneiform sign provenance: One entry per distinct cuneiform codepoint used in the document. Each entry: `U+XXXXX <sign> — English gloss (source)`. Example:
```
- U+12097 𒂗 — EN "lord/priest" (ePSD2 sign EN, Noto Sans Cuneiform)
- U+12017 𒀭 — AN "god/heaven" (ePSD2 sign AN / DINGIR determinative, Noto Sans Cuneiform)
```

- [ ] **Step 9: Run consistency tests against the complete document**

```bash
cd /Users/crashy/Development/cuneiformy
pytest tests/test_anomaly_findings_consistency.py -v
```
Expected: all 5 synthetic tests still pass. If your `extract_numeric_claims` catches issues in the real document, you may need to adjust either the claim or the extractor.

Also run a spot-check that every cuneiform codepoint has provenance:
```bash
python3 -c "
from scripts.docs.consistency import extract_cuneiform_codepoints, check_provenance_coverage
from pathlib import Path
md = Path('docs/anomaly_atlas_findings.md')
codepoints = extract_cuneiform_codepoints(md)
print(f'{len(codepoints)} distinct cuneiform codepoints')
result = check_provenance_coverage(md, codepoints)
print(f'covered: {len(result[\"covered\"])}, missing: {len(result[\"missing\"])}')
if result['missing']:
    print('MISSING:', [hex(c) for c in result['missing']])
"
```
Expected: `missing: 0`. If any are missing, add them to §12.

- [ ] **Step 10: Commit**

```bash
cd /Users/crashy/Development/cuneiformy
git add docs/anomaly_atlas_findings.md
git commit -m "docs: add anomaly atlas findings interpretive document (~9.5k words)"
```

---

## Task 4: Render PDF + journal + README + final sanity

**Files:**
- Create: `docs/anomaly_atlas_findings.pdf` (generated)
- Modify: `docs/EXPERIMENT_JOURNAL.md`
- Modify: `README.md`

### Setup note

Delivery step. Runs the PDF render, does a visual spot-check, commits the PDF and journal/README updates.

- [ ] **Step 1: Render the PDF**

```bash
cd /Users/crashy/Development/cuneiformy
bash scripts/docs/render_anomaly_pdf.sh 2>&1 | tee /tmp/render_output.txt
```

Expected stdout ends with `[render] Wrote docs/anomaly_atlas_findings.pdf (NNN bytes)`. PDF size typically 500KB-2MB for a ~15-page document with a custom font.

If xelatex fails: read the error log carefully. Common issues:
- Font fallback: cuneiform codepoint outside Noto Sans Cuneiform coverage. Find the offending character by searching the markdown.
- Template syntax: check `docs/templates/cuneiformy-pandoc.tex` compiled. Try pandoc's default template as a debug step.

- [ ] **Step 2: Visual inspection of the PDF**

```bash
open docs/anomaly_atlas_findings.pdf
```

Spot-check (no automation):
- All 6 theme headings render with their cuneiform signs.
- The table of contents lists all 13 sections.
- Case studies' first-mention cuneiform renders correctly.
- Body text uses Noto Sans; monospace blocks (if any) use Noto Sans Mono.
- No blank pages (unless deliberately inserted via `\newpage`).
- Appendix §12 lists all cuneiform signs with their `U+XXXXX` codepoint markers.

If anything is off, fix in the markdown and re-render.

- [ ] **Step 3: Determinism check (light)**

PDF byte-determinism is not expected (pandoc/xelatex embed timestamps in metadata). Instead, spot-check structural properties:

```bash
python3 -c "
import subprocess
out1 = subprocess.check_output(['pdfinfo', 'docs/anomaly_atlas_findings.pdf'])
print(out1.decode())"
```
Expected: Pages field shows a sensible page count (15-22 pages for ~10k words). No automation — just visual confirmation.

- [ ] **Step 4: Full suite regression**

```bash
pytest tests/ --ignore=tests/test_integration.py -q
```
Expected: 166 pass (no regression from prior 161 + 5 consistency tests).

- [ ] **Step 5: Commit PDF + generated-artifact companion**

```bash
cd /Users/crashy/Development/cuneiformy
git add docs/anomaly_atlas_findings.pdf
git commit -m "chore: render anomaly atlas findings PDF"
```

- [ ] **Step 6: Add journal entry**

Insert in `docs/EXPERIMENT_JOURNAL.md` AFTER the preamble's `---` and BEFORE the existing 2026-04-20 anomaly atlas entry. Replace every bracketed placeholder with real content from the document you just wrote.

```markdown
## 2026-04-20 — Anomaly Atlas Interpretive Document shipped

**Hypothesis:** The anomaly atlas produced 300+ ranked anomalies but no interpretive prose. Extracting 15-20 most striking findings across six themes into a narrative document (markdown + PDF with embedded cuneiform font) produces a second shareable research artifact — distinct from the cosmogony (five hand-picked concepts) and complementary to the atlas (raw data).

**Method:** New build pipeline (pandoc + xelatex + Noto Sans Cuneiform font committed at `docs/fonts/`, template at `docs/templates/cuneiformy-pandoc.tex`, script at `scripts/docs/render_anomaly_pdf.sh`). Document at `docs/anomaly_atlas_findings.md` (~9,500 words, 13 sections). Six themes: (1) translation failures, (2) specialized cultic vocabulary, (3) grammatical bridges, (4) transliteration shadows, (5) the numeric tail, (6) reading through the floor. Consistency tests in `tests/test_anomaly_findings_consistency.py` verify every numeric claim traces to `docs/anomaly_atlas.json` (pinned commit `ff94533`) and every cuneiform codepoint is documented in §12 provenance appendix.

**Result:** [HEADLINE FINDING 1 — e.g., the en/jizzal/rin2 translation-failure pattern or a specific phrase from §3]. [HEADLINE FINDING 2 — from §5 grammatical bridges or §8 meta]. Document's standalone frame — it does not re-engage with the cosmogony's thesis; the two documents are independent research artifacts.

**Takeaway:** [2-3 sentences on what the document ACTUALLY concludes. Typical: the atlas is more valuable as a diagnostic than as a ranker; the top-1 row of each lens is rarely the most interesting; the middle ranks are where interpretation lives. The cuneiform font infrastructure is reusable — future research documents can adopt the same pandoc template.].

**Artifacts / commits:** `docs/anomaly_atlas_findings.md`, `docs/anomaly_atlas_findings.pdf`, `docs/fonts/NotoSansCuneiform-Regular.ttf`, `docs/templates/cuneiformy-pandoc.tex`, `scripts/docs/render_anomaly_pdf.sh`, `scripts/docs/consistency.py`, `tests/test_anomaly_findings_consistency.py`. Spec: `docs/superpowers/specs/2026-04-20-anomaly-atlas-interpretive-document-design.md`. Plan: `docs/superpowers/plans/2026-04-20-anomaly-atlas-interpretive-document.md`.
```

Fill in the 3 bracketed placeholders with real content BEFORE committing.

- [ ] **Step 7: Add README link**

Update `README.md`'s Research Progress section. Add as the new first bullet under "Recent findings (newest first):":

```markdown
- **2026-04-20 — Anomaly Atlas Interpretive Findings:** Standalone ~9,500-word document + PDF with embedded cuneiform font, surfacing 15-20 atlas findings across six themes. See [`docs/anomaly_atlas_findings.md`](docs/anomaly_atlas_findings.md) (markdown) / [`docs/anomaly_atlas_findings.pdf`](docs/anomaly_atlas_findings.pdf) (PDF with cuneiform).
```

- [ ] **Step 8: Commit journal + README**

```bash
cd /Users/crashy/Development/cuneiformy
git add docs/EXPERIMENT_JOURNAL.md README.md
git commit -m "docs: journal anomaly findings document + link from README"
```

---

## Self-Review

**Spec coverage:**
- Markdown document + all 13 sections → Task 3 Steps 1-8.
- PDF build pipeline (font + template + script) → Task 1 Steps 1-3.
- 2 consistency tests (numeric claims + cuneiform provenance) → Task 2.
- PDF render → Task 4 Step 1.
- Journal entry → Task 4 Step 6.
- README link → Task 4 Step 7.
- Pinned commit (`ff94533`) named in §10 → Task 3 Step 8.
- Cuneiform signs bounded to ~20-25 via headings + case-study first-mentions → Task 3 per-section guidance.

**Placeholder scan:**
- Task 3 per-section targets contain prose-guidance placeholders (descriptions of what to write, not literal prose). This is intentional — the writer produces the prose from the guidance plus the atlas data. No "TODO", "TBD", or "fill in details" in any task step.
- Task 4 Step 6 journal template has 3 bracketed placeholders that explicitly require filling before committing. Those are necessary because the headline findings depend on what the document ends up saying.
- No "Similar to Task N", no "handle edge cases", no "add appropriate error handling" patterns.

**Type consistency:**
- Function signatures in Task 2 (`extract_numeric_claims`, `extract_cuneiform_codepoints`, `find_claim_in_atlas`, `check_provenance_coverage`) are consistent between the test and the implementation.
- Paths consistent: `docs/anomaly_atlas_findings.md`, `docs/anomaly_atlas_findings.pdf`, `docs/fonts/NotoSansCuneiform-Regular.ttf`, `scripts/docs/render_anomaly_pdf.sh`, `docs/templates/cuneiformy-pandoc.tex`, `tests/test_anomaly_findings_consistency.py`, `scripts/docs/consistency.py`.
- Atlas data paths (`docs/anomaly_atlas.json`) consistent across tasks.
- Pinned commit `ff94533` referenced consistently in Task 3 Step 8, Task 4 Step 6.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-20-anomaly-atlas-interpretive-document.md`. Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task with two-stage review. Tasks 1-2 are mechanical infrastructure + TDD; Task 3 is the ~9,500-word prose task (cosmogony-pattern); Task 4 is the PDF render + delivery.

**2. Inline Execution** — batched via `superpowers:executing-plans`.

User pre-selected **option 1** while this plan was being written — proceeding with subagent-driven execution immediately.
