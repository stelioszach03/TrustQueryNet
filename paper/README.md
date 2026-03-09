# TrustQueryNet Paper Sources

This directory contains the paper-facing LaTeX sources for the corrected TrustQueryNet study.

## Files

- `trustquerynet_paper.tex`
  - full manuscript source
- `trustquerynet_abstract.tex`
  - standalone abstract source
- `references.bib`
  - bibliography used by the manuscript
- `references_used.md`
  - plain-language list of the references currently grounding the paper
- `build_pdfs.sh`
  - reproducible local build script

## Outputs

After building, this directory should contain:

- `trustquerynet_paper.pdf`
- `trustquerynet_abstract.pdf`

## Build

```bash
bash paper/build_pdfs.sh
```
