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
