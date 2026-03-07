# TrustQueryNet Supplementary Appendix

## A. Reproducibility Summary

The final paper-facing package is built around persisted artifacts:

- lesion-level split manifests
- noise manifests
- per-run config snapshots
- explicit checkpoint-policy selection
- per-seed metrics
- aggregate multi-seed summaries
- dense selective summaries including AURC
- external-validation tables and plots
- overlap-audit outputs

The corrected final evidence suite uses the `q1_*` config family and the `final-e12` runs generated from the selected `12`-epoch ConvNeXt-Tiny recipe.

## B. Final Run Definitions

### B.1 Internal main comparison

- repair:
  [configs/q1_ham10000_convnext_repair.yaml](../configs/q1_ham10000_convnext_repair.yaml)
- no repair:
  [configs/q1_ham10000_convnext_no_repair.yaml](../configs/q1_ham10000_convnext_no_repair.yaml)
- random repair:
  [configs/q1_ham10000_convnext_random_repair.yaml](../configs/q1_ham10000_convnext_random_repair.yaml)

Final multiseed run names:

- `q1-ham10000-convnext-repair-final-e12-multiseed`
- `q1-ham10000-convnext-no-repair-final-e12-multiseed`
- `q1-ham10000-convnext-random-repair-final-e12-multiseed`

### B.2 Noisy-label anchors

- clean upper:
  [configs/q1_ham10000_convnext_clean_upper.yaml](../configs/q1_ham10000_convnext_clean_upper.yaml)
- GCE no repair:
  [configs/q1_ham10000_convnext_gce_no_repair.yaml](../configs/q1_ham10000_convnext_gce_no_repair.yaml)

Final multiseed run names:

- `q1-ham10000-convnext-clean-upper-final-e12-multiseed`
- `q1-ham10000-convnext-gce-no-repair-final-e12-multiseed`

### B.3 External validation

Final external run names:

- `q1-ham10000-convnext-repair-final-e12-multiseed-external-isic2019-test`
- `q1-ham10000-convnext-no-repair-final-e12-multiseed-external-isic2019-test`
- `q1-ham10000-convnext-random-repair-final-e12-multiseed-external-isic2019-test`

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

## D. Overlap Audit

Overlap audit output:

- HAM10000 samples: `10015`
- ISIC 2019 mapped external samples: `6191`
- exact duplicate images: `0`
- near-duplicate candidate pairs at `dHash <= 4`: `14185`

Interpretation:

- `0` exact matches is the primary integrity result
- the perceptual-hash candidate count is a coarse screen and must not be presented as confirmed leakage

## E. Final Reproduction Commands

### E.1 Internal multiseed evidence

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

### E.2 External validation

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

### E.3 Overlap audit

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

## F. Data, Ethics, and Scope

- This is a retrospective dataset study, not a clinical deployment system.
- The trusted-label repair protocol is simulated oracle supervision rather than real-time expert relabeling.
- External validation reveals substantial residual brittleness under shift.
- The paper’s value is rigorous evaluation, not algorithmic novelty.
