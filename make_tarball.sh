#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
TARBALLS_DIR="$REPO_DIR/tarballs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
TARBALL="$TARBALLS_DIR/thesis_source_$TIMESTAMP.tar.gz"

mkdir -p "$TARBALLS_DIR"

tar -czf "$TARBALL" \
  -C "$REPO_DIR" \
  caltech_thesis.cls \
  nelli_kyle_2026_thesis.tex \
  refs.bib \
  caltech.png \
  chapters/1_intro.tex \
  chapters/2_nelli.tex \
  chapters/3_lovelace_nelli.tex \
  chapters/4_cross_code_cce.tex \
  images/ \
  NelliKyle2026Thesis.pdf

echo "Created: $TARBALL"

# --- self-contained build test ---
# Pass --test to verify the tarball builds the PDF from scratch.
if [[ "${1:-}" == "--test" ]]; then
  WORK_DIR="$(mktemp -d)"
  echo "Testing tarball in $WORK_DIR ..."
  tar -xzf "$TARBALL" -C "$WORK_DIR"

  BUILD_LOG="$WORK_DIR/latexmk.log"
  latexmk -pdf -cd -interaction=nonstopmode \
    -jobname=NelliKyle2026Thesis \
    "$WORK_DIR/nelli_kyle_2026_thesis.tex" \
    > "$BUILD_LOG" 2>&1

  OUTPUT_PDF="$WORK_DIR/NelliKyle2026Thesis.pdf"
  if [[ -f "$OUTPUT_PDF" ]]; then
    echo "Test passed: PDF built successfully."
  else
    echo "Test FAILED: PDF not produced." >&2
  fi
  echo "PDF path: $OUTPUT_PDF"
  echo "Build log: $BUILD_LOG"

  [[ -f "$OUTPUT_PDF" ]] || exit 1
fi
