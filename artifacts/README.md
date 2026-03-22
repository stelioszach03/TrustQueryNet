# Tracked Artifact Summaries

This directory is mostly treated as generated output, but a small paper-facing subset is intentionally retained so the public repository stays understandable and reproducible.

## Intentionally tracked

- `final_evidence/`
  - lightweight multiseed summaries for the final internal and external evidence slice
  - includes aggregate summaries, seed summaries, manifests, and resolved configs
- `paper_tables/`
  - exported comparison tables used by the paper package
- `paper_figures/figure_manifest.json`
  - provenance record for the representative manuscript figures
- `paper_figures/overlap_report.json`
  - paper-facing copy of the overlap summary used in the manuscript package
- `overlap/ham10000-isic2019/overlap_report.json`
  - exact-match and perceptual-hash screening summary

## Intentionally not tracked

- checkpoints
- heavy exploratory run folders
- browser-download export bundles and ZIP packages
- raw datasets
- large local scratch outputs

## Why this split exists

The repo is meant to support a public paper-facing GitHub snapshot without turning `artifacts/` into a dump of heavyweight local outputs. The tracked subset exists to preserve the exact final evidence summaries that support the manuscript, while larger generated files remain ignored.
