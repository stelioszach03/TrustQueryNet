# TrustQueryNet Supplementary Appendix

## A. Scope of This Supplement

This appendix documents the corrected paper-facing evidence slice for TrustQueryNet as an externally validated trustworthy-ML study of dermatoscopic classification under simulated class-dependent label corruption and **budgeted trusted-label repair under simulated oracle supervision**.

The supplement is intentionally narrow. It documents:

- the exact final configs used for the main paper evidence
- the locked seed lists and checkpoint policy
- the final internal and external result summaries
- the paired seed-level statistical comparison layer
- the overlap-audit interpretation
- the commands needed to regenerate the paper-facing tables and figures

It does **not** claim exhaustive statistical testing for every metric, per-class comparison, or exploratory run.

## B. Final Evidence Slice

### B.1 Main configs

Internal main comparison configs:

- repair: [configs/q1_ham10000_convnext_repair.yaml](../configs/q1_ham10000_convnext_repair.yaml)
- no repair: [configs/q1_ham10000_convnext_no_repair.yaml](../configs/q1_ham10000_convnext_no_repair.yaml)
- random repair: [configs/q1_ham10000_convnext_random_repair.yaml](../configs/q1_ham10000_convnext_random_repair.yaml)

Noisy-label anchor configs:

- clean upper: [configs/q1_ham10000_convnext_clean_upper.yaml](../configs/q1_ham10000_convnext_clean_upper.yaml)
- GCE no repair: [configs/q1_ham10000_convnext_gce_no_repair.yaml](../configs/q1_ham10000_convnext_gce_no_repair.yaml)

### B.2 Locked run names

Internal multiseed runs:

- `q1-ham10000-convnext-repair-final-e12-multiseed`
- `q1-ham10000-convnext-no-repair-final-e12-multiseed`
- `q1-ham10000-convnext-random-repair-final-e12-multiseed`
- `q1-ham10000-convnext-clean-upper-final-e12-multiseed`
- `q1-ham10000-convnext-gce-no-repair-final-e12-multiseed`

External multiseed runs:

- `q1-ham10000-convnext-repair-final-e12-multiseed-external-isic2019-test`
- `q1-ham10000-convnext-no-repair-final-e12-multiseed-external-isic2019-test`
- `q1-ham10000-convnext-random-repair-final-e12-multiseed-external-isic2019-test`

### B.3 Locked seeds

Main five-seed comparisons:

- `42`
- `52`
- `62`
- `72`
- `82`

Three-seed anchors:

- `42`
- `52`
- `62`

### B.4 Locked training and evaluation policy

The paper-facing recipe is the selected `e12` ConvNeXt-Tiny slice:

- `12` training epochs
- cross-entropy with `0.05` label smoothing for the main repair/no-repair/random-repair baselines
- explicit checkpoint selection by validation macro-F1
- temperature scaling on internal validation data
- dense selective reporting including AURC, coverage at confidence `0.5`, and risk at confidence `0.5`

The checkpoint policy used for all corrected paper-facing evaluations, including external validation, is:

- `best_val_macro_f1`

## C. Final Result Snapshot

### C.1 Internal main comparison

| Setting | Accuracy | Macro-F1 | ECE | Macro-AUROC | AURC |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Repair` | `0.8350 ± 0.0059` | `0.7152 ± 0.0216` | `0.0445 ± 0.0090` | `0.9382 ± 0.0133` | `0.0728 ± 0.0186` |
| `No repair` | `0.8321 ± 0.0071` | `0.7145 ± 0.0130` | `0.0356 ± 0.0075` | `0.9460 ± 0.0128` | `0.0622 ± 0.0093` |
| `Random repair` | `0.8319 ± 0.0051` | `0.7105 ± 0.0239` | `0.0356 ± 0.0071` | `0.9521 ± 0.0128` | `0.0588 ± 0.0145` |

### C.2 Noisy-label anchors

| Setting | Accuracy | Macro-F1 | ECE | Macro-AUROC | AURC |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Repair` | `0.8350 ± 0.0059` | `0.7152 ± 0.0216` | `0.0445 ± 0.0090` | `0.9382 ± 0.0133` | `0.0728 ± 0.0186` |
| `Clean upper` | `0.8521 ± 0.0055` | `0.7330 ± 0.0288` | `0.0389 ± 0.0176` | `0.9583 ± 0.0076` | `0.0502 ± 0.0095` |
| `GCE no repair` | `0.6774 ± 0.0012` | `0.1224 ± 0.0122` | `0.0268 ± 0.0381` | `0.5782 ± 0.1279` | `0.2399 ± 0.1059` |

### C.3 External main comparison

| Setting | Accuracy | Macro-F1 | ECE | Macro-AUROC | AURC |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Repair external` | `0.5692 ± 0.0145` | `0.4427 ± 0.0117` | `0.2000 ± 0.0253` | `0.8125 ± 0.0180` | `0.2804 ± 0.0299` |
| `No repair external` | `0.5630 ± 0.0078` | `0.4288 ± 0.0123` | `0.2016 ± 0.0123` | `0.8168 ± 0.0154` | `0.2749 ± 0.0183` |
| `Random repair external` | `0.5591 ± 0.0203` | `0.4311 ± 0.0328` | `0.2149 ± 0.0395` | `0.8114 ± 0.0337` | `0.2905 ± 0.0669` |

## D. Paired Statistical Reporting

### D.1 Method

Paper-facing statistical comparisons are computed from the exported `seed_results.csv` files in [artifacts/final_evidence](../artifacts/final_evidence).

The comparison layer is deliberately lightweight and prespecified:

- primary endpoint: calibrated test macro-F1
- secondary endpoint: calibrated test accuracy
- comparison families:
  - repair vs no repair
  - repair vs random repair
  - repair vs GCE no repair
  - external repair vs external no repair
  - external repair vs external random repair

For each paired comparison, we report:

- the observed mean delta across shared seeds
- a paired bootstrap confidence interval for the mean delta
- a two-sided exact sign-flip permutation p-value

Why this design:

- it uses the locked final evidence slice rather than exploratory artifacts
- it is valid for paired multiseed summaries without introducing heavier statistical machinery
- it keeps the paper focused on effect size and uncertainty, not on thresholded p-value hunting

Interpretation note:

- with `5` shared seeds, the minimum attainable two-sided exact sign-flip p-value is `0.0625`
- with `3` shared seeds, the minimum attainable two-sided exact sign-flip p-value is `0.25`

The GCE comparison therefore has a clearly separated effect-size interval but coarse permutation resolution because only the shared three-seed anchor is available.

### D.2 Paired comparison table

| Comparison | Metric | Shared seeds | Delta (repair minus comparator) | 95% paired bootstrap CI | Exact sign-flip p |
| --- | --- | --- | ---: | --- | ---: |
| `Repair vs no repair` | `Internal calibrated macro-F1` | `42,52,62,72,82` | `0.0008` | `[-0.0231, 0.0217]` | `1.0000` |
| `Repair vs random repair` | `Internal calibrated macro-F1` | `42,52,62,72,82` | `0.0045` | `[-0.0112, 0.0262]` | `0.8750` |
| `Repair vs GCE no repair` | `Internal calibrated macro-F1` | `42,52,62` | `0.5918` | `[0.5552, 0.6332]` | `0.2500` |
| `Repair vs no repair` | `Internal calibrated accuracy` | `42,52,62,72,82` | `0.0029` | `[-0.0065, 0.0131]` | `0.6250` |
| `Repair vs random repair` | `Internal calibrated accuracy` | `42,52,62,72,82` | `0.0031` | `[-0.0034, 0.0105]` | `0.6875` |
| `Repair vs GCE no repair` | `Internal calibrated accuracy` | `42,52,62` | `0.1565` | `[0.1515, 0.1636]` | `0.2500` |
| `Repair vs no repair` | `External calibrated macro-F1` | `42,52,62,72,82` | `0.0139` | `[-0.0044, 0.0281]` | `0.1875` |
| `Repair vs random repair` | `External calibrated macro-F1` | `42,52,62,72,82` | `0.0117` | `[-0.0097, 0.0329]` | `0.4375` |
| `Repair vs no repair` | `External calibrated accuracy` | `42,52,62,72,82` | `0.0062` | `[-0.0059, 0.0183]` | `0.3750` |
| `Repair vs random repair` | `External calibrated accuracy` | `42,52,62,72,82` | `0.0102` | `[-0.0087, 0.0290]` | `0.3750` |

Reviewer-facing summary:

- the main repair claims remain directionally favorable but modest
- internal repair does **not** show a reliable seed-paired advantage over no repair or random repair on calibrated macro-F1
- the external repair deltas are also modest and their confidence intervals cross zero
- the repair vs GCE gap is large, but the associated exact p-value is coarse because the shared anchor subset has only three seeds

The machine-readable export for this section lives at:

- [artifacts/paper_tables/significance/paired_significance.md](../artifacts/paper_tables/significance/paired_significance.md)
- [artifacts/paper_tables/significance/paired_significance.csv](../artifacts/paper_tables/significance/paired_significance.csv)
- [artifacts/paper_tables/significance/paired_significance.json](../artifacts/paper_tables/significance/paired_significance.json)

## E. Figure Provenance

The manuscript figures are built from final corrected artifacts rather than exploratory runs.

Representative seed policy:

- choose the repair seed whose calibrated macro-F1 is closest to the repair multiseed mean

Final representative seed:

- internal repair figures: `seed-72`
- external repair figures: `seed-72`

Figure provenance is recorded in:

- [paper/paper_figures/figure_manifest.json](../paper/paper_figures/figure_manifest.json)
- [artifacts/paper_figures/figure_manifest.json](../artifacts/paper_figures/figure_manifest.json)

These figures are qualitative illustrations of calibration and risk-coverage behavior. They are not the basis of the paired statistical comparisons above.

## F. Overlap Audit

Overlap audit output:

- HAM10000 samples: `10015`
- mapped ISIC 2019 external samples: `6191`
- exact duplicate images: `0`
- perceptual-hash candidate pairs at `dHash <= 4`: `14185`

Interpretation:

- `0` exact matches is the primary integrity result
- the perceptual-hash candidate list is a coarse screen for manual or secondary review
- the candidate count must **not** be presented as confirmed overlap or confirmed leakage

Repo artifact:

- [artifacts/overlap/ham10000-isic2019/overlap_report.json](../artifacts/overlap/ham10000-isic2019/overlap_report.json)

## G. Reproduction Commands

### G.1 Internal multiseed evidence

```bash
python scripts/run_multiseed_experiment.py \
  --config configs/q1_ham10000_convnext_repair.yaml \
  --seeds 42 52 62 72 82 \
  --resume-existing
```

```bash
python scripts/run_multiseed_experiment.py \
  --config configs/q1_ham10000_convnext_no_repair.yaml \
  --seeds 42 52 62 72 82 \
  --resume-existing
```

```bash
python scripts/run_multiseed_experiment.py \
  --config configs/q1_ham10000_convnext_random_repair.yaml \
  --seeds 42 52 62 72 82 \
  --resume-existing
```

```bash
python scripts/run_multiseed_experiment.py \
  --config configs/q1_ham10000_convnext_clean_upper.yaml \
  --seeds 42 52 62 \
  --resume-existing
```

```bash
python scripts/run_multiseed_experiment.py \
  --config configs/q1_ham10000_convnext_gce_no_repair.yaml \
  --seeds 42 52 62 \
  --resume-existing
```

### G.2 External validation

```bash
python scripts/run_external_validation.py \
  --multiseed-run-dir artifacts/final_evidence/q1-ham10000-convnext-repair-final-e12-multiseed \
  --ground-truth-csv data/isic2019_external_test/ISIC_2019_Test_GroundTruth.csv \
  --metadata-csv data/isic2019_external_test/ISIC_2019_Test_Metadata.csv \
  --image-dir data/isic2019_external_test/images \
  --num-workers 0
```

```bash
python scripts/run_external_validation.py \
  --multiseed-run-dir artifacts/final_evidence/q1-ham10000-convnext-no-repair-final-e12-multiseed \
  --ground-truth-csv data/isic2019_external_test/ISIC_2019_Test_GroundTruth.csv \
  --metadata-csv data/isic2019_external_test/ISIC_2019_Test_Metadata.csv \
  --image-dir data/isic2019_external_test/images \
  --num-workers 0
```

```bash
python scripts/run_external_validation.py \
  --multiseed-run-dir artifacts/final_evidence/q1-ham10000-convnext-random-repair-final-e12-multiseed \
  --ground-truth-csv data/isic2019_external_test/ISIC_2019_Test_GroundTruth.csv \
  --metadata-csv data/isic2019_external_test/ISIC_2019_Test_Metadata.csv \
  --image-dir data/isic2019_external_test/images \
  --num-workers 0
```

### G.3 Paper-facing significance export

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

### G.4 Overlap audit

```bash
python scripts/audit_ham10000_isic2019_overlap.py \
  --ham-metadata-csv data/ham10000/HAM10000_metadata.csv \
  --ham-image-dir data/ham10000/images \
  --ham-split-csv data/ham10000/splits.csv \
  --isic-ground-truth-csv data/isic2019_external_test/ISIC_2019_Test_GroundTruth.csv \
  --isic-metadata-csv data/isic2019_external_test/ISIC_2019_Test_Metadata.csv \
  --isic-image-dir data/isic2019_external_test/images \
  --output-dir artifacts/overlap/ham10000-isic2019
```

## H. Scope and Ethics

- This is a retrospective dataset study, not a clinical deployment system.
- The trusted-label repair protocol is simulated oracle supervision rather than real-time expert relabeling.
- External validation reveals substantial residual brittleness under shift.
- The paper's contribution is disciplined comparative evidence, not algorithmic novelty or deployment readiness.
