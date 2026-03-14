# TrustQueryNet Paper Package Manifest

This package is the paper-facing documentation set for the corrected Q1-oriented TrustQueryNet study.

## Included files

- [paper_abstract.md](./paper_abstract.md)
  - submission-ready abstract text
- [mini_paper_draft.md](./mini_paper_draft.md)
  - working full manuscript draft with final verified results
- [final_results_tables.md](./final_results_tables.md)
  - paper-ready result tables in one place
- [supplementary_appendix.md](./supplementary_appendix.md)
  - reproducibility, run definitions, overlap summary, and commands
- [portfolio_copy.md](./portfolio_copy.md)
  - condensed portfolio / application variants
- [../README.md](../README.md)
  - repo-level summary aligned with the final evidence slice
- [../paper/trustquerynet_abstract.pdf](../paper/trustquerynet_abstract.pdf)
  - standalone abstract PDF
- [../paper/trustquerynet_paper.pdf](../paper/trustquerynet_paper.pdf)
  - compiled manuscript PDF
- [../paper/references.bib](../paper/references.bib)
  - bibliography used by the LaTeX manuscript
- [../paper/references_used.md](../paper/references_used.md)
  - plain-language checklist of the references currently grounding the paper
- [../paper/trustquerynet_paper.tex](../paper/trustquerynet_paper.tex)
  - full LaTeX manuscript source
- [../paper/paper_figures/internal_external_summary.png](../paper/paper_figures/internal_external_summary.png)
  - compact internal-vs-external summary figure
- [../paper/paper_figures/internal_reliability_calibrated.png](../paper/paper_figures/internal_reliability_calibrated.png)
  - representative internal calibrated reliability plot
- [../paper/paper_figures/external_reliability_calibrated.png](../paper/paper_figures/external_reliability_calibrated.png)
  - representative external calibrated reliability plot
- [../paper/paper_figures/figure_manifest.json](../paper/paper_figures/figure_manifest.json)
  - figure provenance and representative-seed metadata

## Current paper position

Recommended positioning:

- externally validated trustworthy-ML study
- dermatoscopic classification under simulated class-dependent label corruption
- budgeted trusted-label repair under simulated oracle supervision
- calibration and selective prediction under internal and external shift

Recommended lead claim:

- internal gains do not guarantee external trustworthiness

## Current venue strategy

- primary realistic target: Biomedical Signal Processing and Control
- ambitious target: IEEE Journal of Biomedical and Health Informatics
- fallback: BMC Medical Imaging

## Packaging note

This manifest covers the repo-side paper documents. The larger experiment artifact bundle is produced separately with `scripts/export_results_bundle.py`.
