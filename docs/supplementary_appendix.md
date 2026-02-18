# TrustQueryNet Supplementary Appendix

> This appendix currently mixes verified exploratory results with the corrected Q1 rerun scaffolding. The `q1_*.yaml` configs are the canonical rerun entrypoints after the new integrity fixes.

## A. Reproducibility summary

The reproducible package is built around persisted artifacts:

- lesion-level split manifests
- noise manifests
- per-run config snapshots
- explicit checkpoint-policy selection
- per-seed metrics
- aggregate multi-seed summaries
- ablation tables
- dense selective summaries including AURC
- external-validation tables and plots
- overlap-audit reports

## B. Main run definitions

### B.1 Exploratory internal model

- config: [configs/full_ham10000_convnext_no_weighted.yaml](../configs/full_ham10000_convnext_no_weighted.yaml)
- multi-seed run directory:
  `artifacts/runs/full-ham10000-convnext-no-weighted-multiseed`

### B.2 Corrected Q1 rerun configs

- repair:
  [configs/q1_ham10000_convnext_repair.yaml](../configs/q1_ham10000_convnext_repair.yaml)
- no repair:
  [configs/q1_ham10000_convnext_no_repair.yaml](../configs/q1_ham10000_convnext_no_repair.yaml)
- random repair:
  [configs/q1_ham10000_convnext_random_repair.yaml](../configs/q1_ham10000_convnext_random_repair.yaml)
- clean-label upper bound:
  [configs/q1_ham10000_convnext_clean_upper.yaml](../configs/q1_ham10000_convnext_clean_upper.yaml)
- robust-loss baseline:
  [configs/q1_ham10000_convnext_gce_no_repair.yaml](../configs/q1_ham10000_convnext_gce_no_repair.yaml)
- weighted secondary ablation:
  [configs/q1_ham10000_convnext_weighted_secondary.yaml](../configs/q1_ham10000_convnext_weighted_secondary.yaml)

### B.3 External validation

- dataset: official ISIC 2019 test set
- evaluation entrypoint:
  [scripts/run_external_validation.py](../scripts/run_external_validation.py)

## C. Reproduction commands

### C.1 Corrected internal repair benchmark

```bash
python scripts/run_multiseed_experiment.py \
  --config configs/q1_ham10000_convnext_repair.yaml \
  --seeds 42 52 62 72 82 \
  --resume-existing
```

### C.2 Repair baselines

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

### C.3 Required anchors

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

### C.4 Secondary weighted comparison

```bash
python scripts/run_multiseed_experiment.py \
  --config configs/q1_ham10000_convnext_weighted_secondary.yaml \
  --seeds 42 52 62 72 82 \
  --resume-existing
```

### C.5 External validation

```bash
python scripts/download_isic2019_external.py \
  --output-root data/isic2019_external_test
```

```bash
python scripts/run_external_validation.py \
  --multiseed-run-dir artifacts/runs/q1-ham10000-convnext-repair-multiseed \
  --ground-truth-csv data/isic2019_external_test/ISIC_2019_Test_GroundTruth.csv \
  --metadata-csv data/isic2019_external_test/ISIC_2019_Test_Metadata.csv \
  --image-dir data/isic2019_external_test/images \
  --num-workers 0
```

### C.6 Overlap audit

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

## D. Label mapping for ISIC 2019

The external dataset uses challenge labels that do not perfectly match the HAM10000 taxonomy. The current mapping is:

- `MEL -> mel`
- `NV -> nv`
- `BCC -> bcc`
- `AK -> akiec`
- `SCC -> akiec`
- `BKL -> bkl`
- `DF -> df`
- `VASC -> vasc`
- `UNK -> filtered`

This preserves the current `7`-class internal training space without introducing an unsupported extra class during external evaluation.

## E. Data statements

- HAM10000 is used as the primary development dataset.
- ISIC 2019 test is used for external validation.
- Both were downloaded from official ISIC challenge / archive endpoints.
- The current project is intended for research and portfolio use, not clinical deployment.

## F. Ethics and limitation statements

Recommended wording:

- This study evaluates retrospective dermatoscopic datasets and does not constitute a clinical decision-support system.
- The trusted-label repair protocol is simulated oracle supervision rather than prospective human annotation.
- External validation reveals a substantial performance drop under dataset shift.
- Internal temperature scaling does not reliably improve calibration on the external test set.
- No prospective, reader-study, or clinician-in-the-loop evaluation is included.
- The current study emphasizes reproducibility and honest reporting over claiming deployment readiness.
