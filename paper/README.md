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
- `paper_figures/`
  - exported internal/external paper figures and figure manifest
- `build_pdfs.sh`
  - reproducible local build script
- `../artifacts/final_evidence/`
  - lightweight final multiseed summaries used for paper-facing tables
- `../artifacts/paper_tables/significance/paired_significance.md`
  - paired seed-level statistical comparison table used by the supplement
- `../artifacts/README.md`
  - explains which artifact summaries are intentionally tracked in git and which heavier outputs remain ignored

## Outputs

After building, this directory should contain:

- `trustquerynet_paper.pdf`
- `trustquerynet_abstract.pdf`

## Build

```bash
bash paper/build_pdfs.sh
```

To regenerate the paper-facing paired statistical comparison table before rebuilding:

```bash
python scripts/export_significance_table.py \
  --runs-root artifacts/final_evidence \
  --repair-run q1-ham10000-convnext-repair-final-e12-multiseed \
  --no-repair-run q1-ham10000-convnext-no-repair-final-e12-multiseed \
  --random-repair-run q1-ham10000-convnext-random-repair-final-e12-multiseed \
  --gce-run q1-ham10000-convnext-gce-no-repair-final-e12-multiseed \
  --repair-external-run q1-ham10000-convnext-repair-final-e12-multiseed-external-isic2019-test \
  --no-repair-external-run q1-ham10000-convnext-no-repair-final-e12-multiseed-external-isic2019-test \
  --random-repair-external-run q1-ham10000-convnext-random-repair-final-e12-multiseed-external-isic2019-test \
  --output-dir artifacts/paper_tables/significance
```
