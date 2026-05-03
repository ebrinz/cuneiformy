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
  --variable=mainfont:"Palatino" \
  --variable=monofont:"Courier" \
  --number-sections \
  --toc \
  --toc-depth=2 \
  -o "$OUT"

if [ ! -s "$OUT" ]; then
  echo "ERROR: PDF was not produced or is empty." >&2
  exit 1
fi

echo "[render] Wrote $OUT ($(wc -c < "$OUT") bytes)"
