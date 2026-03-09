#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER_DIR="$ROOT_DIR/paper"

cd "$PAPER_DIR"

latexmk -pdf -interaction=nonstopmode trustquerynet_abstract.tex
latexmk -pdf -interaction=nonstopmode trustquerynet_paper.tex

# Keep the generated PDFs, but remove build intermediates.
latexmk -c trustquerynet_abstract.tex
latexmk -c trustquerynet_paper.tex
