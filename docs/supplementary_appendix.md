# TrustQueryNet Supplementary Appendix

## A. Reproducibility summary

The final paper-facing package is built around persisted artifacts:

- lesion-level split manifests
- noise manifests
- per-run config snapshots
- per-seed metrics
- aggregate multi-seed summaries
- ablation tables
- external-validation tables and plots

The main internal and external claims are based on `5` seeds each.

## B. Main run definitions

### B.1 Main internal model

- config: [configs/full_ham10000_convnext_no_weighted.yaml](/Users/stelioszacharioudakis/Documents/TrustQueryNet/configs/full_ham10000_convnext_no_weighted.yaml)
- multi-seed run directory:
  `artifacts/runs/full-ham10000-convnext-no-weighted-multiseed`

### B.2 Ablation models

- weighted-sampler comparison:
  [configs/full_ham10000_convnext.yaml](/Users/stelioszacharioudakis/Documents/TrustQueryNet/configs/full_ham10000_convnext.yaml)
- no-query comparison:
  [configs/full_ham10000_convnext_no_querying.yaml](/Users/stelioszacharioudakis/Documents/TrustQueryNet/configs/full_ham10000_convnext_no_querying.yaml)

### B.3 External validation

- dataset: official ISIC 2019 test set
- evaluation entrypoint:
  [run_external_validation.py](/Users/stelioszacharioudakis/Documents/TrustQueryNet/scripts/run_external_validation.py)

## C. Reproduction commands

### C.1 Internal multi-seed benchmark

```bash
python scripts/run_multiseed_experiment.py \
  --config configs/full_ham10000_convnext_no_weighted.yaml \
  --seeds 42 52 62 72 82 \
  --resume-existing
```

### C.2 Ablation study

```bash
python scripts/run_multiseed_experiment.py \
  --config configs/full_ham10000_convnext.yaml \
  --seeds 42 52 62 72 82 \
  --resume-existing
```

```bash
python scripts/run_multiseed_experiment.py \
  --config configs/full_ham10000_convnext_no_querying.yaml \
  --seeds 42 52 62 72 82 \
  --resume-existing
```

```bash
python scripts/export_ablation_table.py \
  --runs-root artifacts/runs \
  --run-spec full-ham10000-convnext-no-weighted-multiseed::Full\ model \
  --run-spec full-ham10000-convnext-balanced-multiseed::Weighted\ sampler \
  --run-spec full-ham10000-convnext-no-querying-multiseed::No\ query\ rounds
```

### C.3 External validation

```bash
python scripts/download_isic2019_external.py \
  --output-root data/isic2019_external_test
```

```bash
python scripts/run_external_validation.py \
  --multiseed-run-dir artifacts/runs/full-ham10000-convnext-no-weighted-multiseed \
  --ground-truth-csv data/isic2019_external_test/ISIC_2019_Test_GroundTruth.csv \
  --metadata-csv data/isic2019_external_test/ISIC_2019_Test_Metadata.csv \
  --image-dir data/isic2019_external_test/images \
  --num-workers 0
```

## D. Label mapping for ISIC 2019

The external dataset uses challenge labels that do not perfectly match the HAM10000 taxonomy. The current verified mapping is:

- `MEL -> mel`
- `NV -> nv`
- `BCC -> bcc`
- `AK -> akiec`
- `SCC -> akiec`
- `BKL -> bkl`
- `DF -> df`
- `VASC -> vasc`
- `UNK -> filtered`

This mapping preserves the current `7`-class internal training space without introducing an unsupported extra class during external evaluation.

## E. Data statements

- HAM10000 is used as the primary development dataset.
- ISIC 2019 test is used only for external validation.
- Both were downloaded from official ISIC challenge / archive endpoints.
- The current project is intended for research and portfolio use, not clinical deployment.

## F. Ethics and limitation statements

Recommended paper wording:

- This study evaluates retrospective dermatoscopic datasets and does not constitute a clinical decision-support system.
- External validation reveals a substantial performance drop under dataset shift.
- Internal temperature scaling does not reliably improve calibration on the external test set.
- No prospective, reader-study, or clinician-in-the-loop evaluation is included.
- The current study emphasizes reproducibility and honest reporting over claiming deployment readiness.
