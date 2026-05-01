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
