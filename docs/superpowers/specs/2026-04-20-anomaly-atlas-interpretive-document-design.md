# Anomaly Atlas Interpretive Document (markdown + PDF with cuneiform font)

**Date:** 2026-04-20
**Status:** Approved (brainstorming), pending writing-plans
**Branch:** `master` (to be cut into a fresh feature branch at implementation time)
**Follows:** `docs/superpowers/specs/2026-04-20-anomaly-atlas-design.md`
**Journal:** `docs/EXPERIMENT_JOURNAL.md`

## Summary

Produce `docs/anomaly_atlas_findings.md` (~9,000-10,000 words) plus a derived
`docs/anomaly_atlas_findings.pdf` rendered via pandoc + xelatex with embedded
Noto Sans Cuneiform font. The document surfaces the most interesting findings
from the 2026-04-20 atlas through six thematic sections: translation failures,
specialized cultic vocabulary, grammatical bridges, transliteration shadows,
the numeric tail, and a meta-essay on reading the atlas honestly. Key Sumerian
terms appear in actual cuneiform signs alongside transliterations; the PDF
build embeds the font so readers see cuneiform regardless of system fonts.

Treated as a standalone document — does NOT re-engage with the earlier
cosmogony document's thesis. The cosmogony and this anomaly document are
independent research artifacts.

## Motivation

The 2026-04-20 anomaly atlas (six lenses over 35,508 aligned Sumerian tokens)
produced 300+ ranked anomalies but no interpretive prose. This document
extracts the most striking 15-20 findings and gives each a ~500-600 word
vignette grouped by theme. The goal is a second-shippable research artifact
that demonstrates what atlas data actually reveals once an interpretive reader
engages with it — distinct from the cosmogony document (five-concept deep
dive) and complementary to the atlas itself (raw data).

The cuneiform-font rendering is an intentional production-value step: a
research document on ancient Sumerian is strictly improved by showing the
actual script rather than only transliteration. The PDF produces a
consistent reading experience regardless of the reader's system fonts.

## Audience

Same A/B midpoint as the cosmogony document:
- Assyriologists who want rigorous citations, transliteration conventions, and
  ePSD2 sign-mapping provenance.
- AI/NLP/embedding researchers who want to see what semantic alignment at
  scale actually reveals about ancient vocabulary.

Requires vector-space familiarity but not Sumerological background; cuneiform
is treated as decorative (primary identification is transliteration) so
readers without Sumerological training can still follow the prose.

## Scope

### In scope

- New `docs/anomaly_atlas_findings.md` — ~9,000-10,000 words, 13 sections.
- Derived `docs/anomaly_atlas_findings.pdf` — pandoc + xelatex build.
- New `scripts/docs/render_anomaly_pdf.sh` — PDF build script with preflight checks.
- New `docs/templates/cuneiformy-pandoc.tex` — pandoc LaTeX template with cuneiform-font auto-switching.
- New `docs/fonts/NotoSansCuneiform-Regular.ttf` — Noto Sans Cuneiform (OFL-licensed, ~1.5MB).
- New `tests/test_anomaly_findings_consistency.py` — 2 consistency tests.
- Modified `docs/EXPERIMENT_JOURNAL.md` — journal entry.

### Out of scope

- Revisiting the cosmogony document's thesis. Standalone document per user choice (option A in brainstorming).
- Interactive visualizations / web dashboards.
- CI integration for PDF regeneration — manual rebuild only.
- Cross-civilizational comparison — `hyper-glyphy` monorepo reorg + Egyptian sibling are queued workstreams.
- Exhaustive sign mapping for every token in atlas tables — cuneiform reserved for headings + case-study highlights (~20-25 signs total).
- Peer-reviewed Sumerological accuracy guarantees — signs are curated from ePSD2.
- Accessibility overlays for cuneiform codepoints.
- Hyper-glyphy reorganization (estimated ~1 week; separate workstream).
- Regenerating the cosmogony PDF (the cosmogony document is markdown-only for now; could be a follow-up pattern).

### Deliverables produced

- 1 new document, 1 PDF, 1 build script, 1 pandoc template, 1 committed font file, 1 test file with 2 tests, 1 journal entry.

## Success Criteria

- Markdown committed with all 6 themes populated; no TODOs.
- All 15-20 case studies have anchor source citations and atlas JSON row citations.
- PDF regenerable via `bash scripts/docs/render_anomaly_pdf.sh`. Script passes preflight checks (pandoc, xelatex, font file) and produces a non-empty PDF.
- PDF opens and renders cuneiform correctly on macOS Preview (spot-checked).
- Every cuneiform codepoint in the markdown has a provenance note in §12 appendix.
- Every numeric claim in prose has a path in `docs/anomaly_atlas.json` — sampled verification via the consistency test.
- `tests/test_anomaly_findings_consistency.py` — both tests pass.
- Full test suite: 161+ pass (no regressions on prior 161 + 2 new consistency tests).
- Pinned commit (`ff94533`, the atlas baseline) named in §10 for reproducibility.
- Journal entry with headline findings per theme.

## Design

### File layout

```
docs/
  anomaly_atlas_findings.md                NEW — ~9,000-10,000 words
  anomaly_atlas_findings.pdf               NEW — rendered via pandoc + xelatex
  templates/
    cuneiformy-pandoc.tex                  NEW — pandoc LaTeX template
  fonts/
    NotoSansCuneiform-Regular.ttf          NEW — embedded font (OFL license)

scripts/docs/                              NEW directory
  render_anomaly_pdf.sh                    NEW — build script with preflight checks

tests/
  test_anomaly_findings_consistency.py     NEW — 2 consistency tests
```

### Document structure

```
§ 0   Abstract                             (~200 words)
§ 1   Introduction                         (~500 words)
§ 2   Methodology                          (~500 words — sketch, link to atlas spec)

§ 3   Theme 1: Translation failures         (~1,800 words, 3 case studies)
         Heading sign: 𒂗 (EN)
         Cases: en → priest / jizzal → ear / rin2 → lord
         
§ 4   Theme 2: Specialized cultic vocabulary (~1,500 words, 3-4 cases)
         Heading sign: 𒄑 (GIŠ)
         Cases: uttu / gidri / ebgal / asag
         
§ 5   Theme 3: Grammatical bridges          (~1,500 words, 3 cases)
         Heading sign: 𒆠 (KI)
         Cases: ningir2su2kake4 / kas4ke4nesze3 / karzidda
         
§ 6   Theme 4: Transliteration shadows      (~800 words)
         Heading sign: 𒊮
         Single discussion of c/š variant pairs in Lens 5
         
§ 7   Theme 5: The numeric tail              (~800 words)
         Heading sign: 𒀸 (AŠ)
         Why Lens 2 is dominated by 1(disz) and related numeric tokens
         
§ 8   Theme 6: Reading through the floor     (~1,000 words — meta)
         Heading sign: 𒄀 (ZI)
         Essay on atlas-reading: top-1 is noise, middle is interpretation
         
§ 9   Synthesis                             (~500 words)
§ 10  Reproducibility                       (~200 words — pinned atlas commit ff94533)
§ 11  References                            (ETCSL text IDs + ePSD2 + secondary scholarship)
§ 12  Appendix: cuneiform sign provenance   (~300 words)
```

Total: ~9,500 words of prose + tables + cuneiform.

### Per-case-study structure (applied in §3-8)

Scaled-down from the cosmogony's 8-section template because the document
covers more ground with less depth per case. Each case study is ~500-600 words
structured as:

1. **The finding** — 1 sentence.
2. **The anchor** — ~100 words on the ePSD2 / Sumerological gloss.
3. **The geometry** — ~200 words on what the atlas data show (top-K neighbors, cosine values, Jaccard distance, bridge score — whichever lens(es) surfaced this token).
4. **Interpretation** — ~200 words of hedged reading.

Each case study's first mention of a Sumerian token uses the format
`𒂗 *en* "lord"` (cuneiform + italic transliteration + English gloss).
Subsequent mentions: just transliteration.

### Cuneiform sign usage

Bounded scope — approximately 20-25 signs total across the document:
- 6 theme headings × 1-2 signs = 6-12 signs
- 15-20 case studies × 1 sign each (on first mention) = 15-20 signs
- Minor duplication across themes = net ~20-25 distinct signs

**Sources for sign mappings:**
- ORACC ePSD2 (http://oracc.museum.upenn.edu/epsd2/) — primary
- Sjöberg, *The Sumerian Dictionary of the University Museum of the University of Pennsylvania* — secondary, for ambiguous cases
- Borger, *Mesopotamisches Zeichenlexikon* — tertiary reference for sign identification

The §12 appendix documents, for each sign used, its source and any ambiguity the author had to adjudicate.

### PDF build pipeline

```
docs/anomaly_atlas_findings.md
         │ (UTF-8 encoded; cuneiform codepoints in Unicode block U+12000–U+123FF)
         ▼
scripts/docs/render_anomaly_pdf.sh
         │ (pandoc + xelatex via template docs/templates/cuneiformy-pandoc.tex)
         │ (font path docs/fonts/NotoSansCuneiform-Regular.ttf)
         ▼
docs/anomaly_atlas_findings.pdf
```

Build command (inside the script):

```bash
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
  -o docs/anomaly_atlas_findings.pdf
```

### Pandoc template (cuneiform auto-switching)

Key additions to pandoc's default LaTeX template:

```latex
\usepackage{fontspec}
\setmainfont{$mainfont$}
\setmonofont{$monofont$}

\newfontfamily{\cuneiformfont}{Noto Sans Cuneiform}[
  Path=./docs/fonts/,
  Extension=.ttf,
  UprightFont=NotoSansCuneiform-Regular
]

% Auto-wrap any character in the Cuneiform Unicode block (U+12000–U+123FF)
% in the cuneiform font. This means markdown prose can embed cuneiform
% directly without manual \cuneiform{...} wrappers.
\XeTeXinterchartoks \XeTeXcharclass`\ \"0 = {}
\XeTeXcharclass "12000 = 4
\XeTeXcharclass "12400 = 0
\XeTeXinterchartoks 0 4 = {\begingroup\cuneiformfont}
\XeTeXinterchartoks 4 0 = {\endgroup}
```

### Build script preflight

`scripts/docs/render_anomaly_pdf.sh` checks:
- `command -v pandoc` — reports required version
- `command -v xelatex` — reports required distribution (MacTeX, TeX Live)
- `test -f docs/fonts/NotoSansCuneiform-Regular.ttf` — reports where to download

Missing any → exit 1 with actionable install instructions.

### System dependencies

Not pip installable. The build script documents these:
- `pandoc` ≥ 3.0 (`brew install pandoc` on macOS, package manager elsewhere)
- `xelatex` (part of MacTeX / TeX Live xetex package)
- Noto Sans Cuneiform (committed as `docs/fonts/NotoSansCuneiform-Regular.ttf`; no system-level install required)

No new Python dependencies.

### Tests

New `tests/test_anomaly_findings_consistency.py`. Both tests parse the markdown file directly:

- **`test_every_numeric_claim_has_json_path`** — scan prose for numbers matching cosine/Jaccard/bridge patterns, verify at least 10-20 distinct numeric claims in the prose cross-reference to a path in `docs/anomaly_atlas.json`. Not exhaustive; catches copy-paste errors.
- **`test_every_cuneiform_sign_has_provenance_note`** — scan for characters in Unicode Cuneiform block; verify each distinct codepoint appears in the §12 provenance appendix.

Tests do NOT regenerate the PDF — PDF regeneration is a manual step before commits.

### Reproducibility

- Pinned atlas commit (`ff94533`) named in §10; all numeric claims in prose are from the atlas JSON at that commit.
- PDF regeneration is a single command; deterministic given the markdown + template + font.
- Build script fails loudly on missing dependencies rather than producing a partial artifact.

## Error Handling

| Condition | Behavior |
|---|---|
| pandoc not installed | Build script prints install instructions and exits 1. No partial PDF produced. |
| xelatex not installed | Same — script exits 1 with macOS/Linux install directions. |
| Noto Sans Cuneiform font missing | Script exits 1 with "run git lfs pull" or similar. |
| Markdown references a cuneiform character not in §12 appendix | Consistency test fails; commit blocked until either the sign is added to appendix or removed from prose. |
| Numeric claim in prose doesn't match any atlas JSON path | Consistency test flags the mismatch; author must fix or add a non-atlas citation. |
| pandoc template syntax error | xelatex fails, script surfaces the LaTeX error log for debugging. |
| Cuneiform codepoint outside U+12000–U+123FF | The auto-switching template won't handle it — use the Sumero-Akkadian Cuneiform block only (U+12000–U+123FF). If future signs fall in the Cuneiform Numbers/Punctuation block (U+12400–U+1247F), extend the template. |

## Testing Strategy

Minimal — this is a prose workstream, not a code workstream.

- **Consistency tests** (2 total, fast):
  - `test_every_numeric_claim_has_json_path` — sampled verification.
  - `test_every_cuneiform_sign_has_provenance_note` — sign/provenance cross-check.
- **Manual verification** before commit:
  - Run `bash scripts/docs/render_anomaly_pdf.sh`, visually inspect PDF:
    - All 6 theme headings render with cuneiform.
    - Case-study first-mentions show cuneiform + italic transliteration.
    - Tables align (no line-wrap issues).
    - TOC present with 2-level depth.
  - Compare PDF length estimate to target (~15-18 pages).

No automated visual regression for the PDF. Acceptable risk — PDF rendering is sensitive to matplotlib/LaTeX version differences and byte-level PDF comparison is brittle.

## Reproducibility

- All numeric claims in prose trace to `docs/anomaly_atlas.json` at pinned commit `ff94533`.
- All cuneiform signs have provenance documented in §12 appendix.
- PDF build is a single command; regeneration produces a fresh PDF from the exact markdown + template + font files in the repo.
- The repository's first PDF artifact — establishes a pattern future research documents can adopt (cosmogony PDF is a candidate follow-up).

## Operational notes

**Success path:** Author writes the markdown in 6 theme sections, running consistency tests after each section. Fixes numeric-claim mismatches or adds missing provenance notes as they arise. After the whole document is drafted, runs `bash scripts/docs/render_anomaly_pdf.sh`, inspects the PDF visually, commits both markdown + PDF. Journal entry added. Merge to master.

**Failure paths:**
- Consistency test fails on numeric claim: author checks the atlas JSON path, fixes the number or adds the correct citation.
- PDF build fails with LaTeX error: author reads the xelatex log (usually font-related or a typo in a `\newcommand`), fixes, re-runs.
- Cuneiform sign doesn't render in PDF (shows as box): either the codepoint isn't in Noto Sans Cuneiform's coverage (check coverage map) or the auto-switching template isn't catching the codepoint range. Diagnose via `xelatex` compile log.

## Follow-up work (out of this spec)

- Regenerate cosmogony PDF using the same template + font infrastructure.
- Hyper-glyphy monorepo reorganization (Phase α, ~1 week) then Egyptian sibling-language build (Phase β, ~2 weeks).
- Additional atlas-derived short essays on specific findings if the top-tier anomalies spawn their own research threads.
- Peer-review outreach — if the document is shared, a Sumerologist's critique of specific sign mappings would improve §12 provenance appendix.
