# TrustQueryNet

TrustQueryNet is a local-first, research-oriented pipeline for dermatoscopic image classification under noisy supervision, **budgeted trusted-label repair under simulated oracle supervision**, calibration, and selective prediction.

The current repo targets HAM10000 as the primary development dataset and ISIC 2019 as the main external validation set.

## Implemented scope

This repository currently includes:

- lesion-level / group-aware HAM10000 splitting with persisted manifests
- symmetric and transition-matrix label noise
- pretrained backbones through `timm`
- cross-entropy, generalized cross-entropy, and symmetric cross-entropy losses
- budgeted trusted-label repair loops with deterministic random-repair support
- temperature scaling for calibration
- dense selective metrics with risk-coverage curves and AURC
- multi-seed aggregation and export tooling
- external validation on the official ISIC 2019 test set
- overlap-audit tooling for HAM10000 vs ISIC 2019

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
pytest -q
python scripts/run_experiment.py --config configs/quick_cifar100.yaml
```

## Verified status

Verified end-to-end:

- local package install in editable mode
- local test suite
- local CIFAR100 smoke runs
- Colab UI pilot HAM10000 run
- Colab UI full HAM10000 run with exported artifacts bundle
- Colab UI multi-seed HAM10000 benchmarking
- Colab UI HAM10000 ablation study
- Colab UI external validation on the official ISIC 2019 test set

Recommended workflow:

- use [notebooks/trustquerynet_local.ipynb](notebooks/trustquerynet_local.ipynb) for local-first exploration
- use [notebooks/trustquerynet_colab.ipynb](notebooks/trustquerynet_colab.ipynb) in browser Colab UI for Drive-backed runs and artifact persistence

## Current verified exploratory results

The numbers below are the **currently verified exploratory results** from the pre-Q1 integrity slice. They are useful development evidence, but they should not be treated as the final publication claims after the corrected reruns.

Main exploratory internal result:

- config: [configs/full_ham10000_convnext_no_weighted.yaml](configs/full_ham10000_convnext_no_weighted.yaml)
- exported multi-seed run: `full-ham10000-convnext-no-weighted-multiseed`
- setup: ConvNeXt-Tiny, cross-entropy with `0.05` label smoothing, transition-matrix noise, entropy-based trusted-label repair, lesion-level fixed split, `5` seeds

Calibrated internal HAM10000 metrics (`mean ± std` across seeds):

| Setting | Accuracy | Macro-F1 | ECE | Macro-AUROC | Coverage@0.5 | Risk@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Exploratory full model (no weighted sampler)` | 0.8454 ± 0.0054 | 0.7265 ± 0.0080 | 0.0375 ± 0.0105 | 0.9599 ± 0.0048 | 0.9467 ± 0.0039 | 0.1303 ± 0.0077 |
| `Exploratory weighted sampler` | 0.8074 ± 0.0095 | 0.6933 ± 0.0244 | 0.0327 ± 0.0053 | 0.9410 ± 0.0096 | 0.9313 ± 0.0095 | 0.1630 ± 0.0101 |
| `Exploratory no-query run` | 0.7989 ± 0.0072 | 0.6742 ± 0.0234 | 0.0380 ± 0.0114 | 0.9327 ± 0.0065 | 0.9250 ± 0.0062 | 0.1668 ± 0.0093 |

Exploratory external validation on the official ISIC 2019 test set:

| External setting | Accuracy | Macro-F1 | ECE | Macro-AUROC | Coverage@0.5 | Risk@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ISIC 2019 external test` | 0.5551 ± 0.0263 | 0.4256 ± 0.0312 | 0.2236 ± 0.0462 | 0.8237 ± 0.0158 | 0.8576 ± 0.0315 | 0.3966 ± 0.0357 |

Why these exploratory results matter:

- repair appears to help relative to the exploratory no-query comparison
- removing the weighted sampler improved internal discrimination in the exploratory slice
- external validation reveals a substantial domain-shift gap
- calibration fitted internally does not transfer reliably under external shift

## Q1 rerun config family

The corrected rerun family for the Q1-oriented paper path now lives in:

- [configs/q1_ham10000_convnext_repair.yaml](configs/q1_ham10000_convnext_repair.yaml)
- [configs/q1_ham10000_convnext_no_repair.yaml](configs/q1_ham10000_convnext_no_repair.yaml)
- [configs/q1_ham10000_convnext_random_repair.yaml](configs/q1_ham10000_convnext_random_repair.yaml)
- [configs/q1_ham10000_convnext_clean_upper.yaml](configs/q1_ham10000_convnext_clean_upper.yaml)
- [configs/q1_ham10000_convnext_gce_no_repair.yaml](configs/q1_ham10000_convnext_gce_no_repair.yaml)
- [configs/q1_ham10000_convnext_weighted_secondary.yaml](configs/q1_ham10000_convnext_weighted_secondary.yaml)

These configs add:

- explicit checkpoint-policy selection
- AMP-enabled training
- longer training with warmup and early stopping hooks
- dense selective metrics / AURC defaults
- deconfounded no-repair baseline
- matched-budget random-repair baseline
- clean-label upper bound
- robust-loss baseline scaffold

Do not overwrite the exploratory run directories when producing corrected publication reruns.

## Multi-seed benchmarking

Run a config across multiple seeds:

```bash
python scripts/run_multiseed_experiment.py \
  --config configs/q1_ham10000_convnext_repair.yaml \
  --seeds 42 52 62 72 82 \
  --resume-existing
```

This writes a sibling directory next to the base run output, for example:

- `artifacts/runs/q1-ham10000-convnext-repair-multiseed/seed-42`
- `artifacts/runs/q1-ham10000-convnext-repair-multiseed/aggregate_results.json`
- `artifacts/runs/q1-ham10000-convnext-repair-multiseed/aggregate_results.md`

On HAM10000, keep `split_csv` populated so the lesion-level split stays fixed across seeds while the training randomness changes.

## Repair and ablation workflow

The Q1-oriented comparison should be built around:

- `q1-ham10000-convnext-repair`
- `q1-ham10000-convnext-random-repair`
- `q1-ham10000-convnext-no-repair`

Secondary comparisons:

- `q1-ham10000-convnext-clean-upper`
- `q1-ham10000-convnext-gce-no-repair`
- `q1-ham10000-convnext-weighted-secondary`

After running each one with the multi-seed runner, export a single paper-ready comparison table:

```bash
python scripts/export_ablation_table.py \
  --runs-root artifacts/runs \
  --run-spec q1-ham10000-convnext-repair-multiseed::Trusted\ repair \
  --run-spec q1-ham10000-convnext-random-repair-multiseed::Random\ repair \
  --run-spec q1-ham10000-convnext-no-repair-multiseed::No\ repair \
  --output-dir artifacts/ablations/ham10000
```

## External validation workflow

The external-validation slice targets the official `ISIC 2019` test set and maps its challenge labels into the HAM10000-style 7-class taxonomy:

- `AK` + `SCC` -> `akiec`
- `MEL` -> `mel`
- `NV` -> `nv`
- `BCC` -> `bcc`
- `BKL` -> `bkl`
- `DF` -> `df`
- `VASC` -> `vasc`
- `UNK` is filtered by default

Download the official ISIC 2019 external test set:

```bash
python scripts/download_isic2019_external.py \
  --output-root /content/drive/MyDrive/ISIC2019_external_test
```

Then evaluate a completed multi-seed run on that external test set:

```bash
python scripts/run_external_validation.py \
  --multiseed-run-dir /content/drive/MyDrive/TrustQueryNet/artifacts/runs/q1-ham10000-convnext-repair-multiseed \
  --ground-truth-csv /content/drive/MyDrive/ISIC2019_external_test/ISIC_2019_Test_GroundTruth.csv \
  --metadata-csv /content/drive/MyDrive/ISIC2019_external_test/ISIC_2019_Test_Metadata.csv \
  --image-dir /content/drive/MyDrive/ISIC2019_external_test/images \
  --num-workers 0
```

To audit potential overlap between HAM10000 and the external ISIC 2019 slice:

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

## Colab workflow

1. Open [notebooks/trustquerynet_colab.ipynb](notebooks/trustquerynet_colab.ipynb) in Colab UI.
2. Mount Drive in the notebook.
3. Pull the latest repo state.
4. Run a smoke test with one of the `q1_*.yaml` configs.
5. Launch the full multi-seed reruns only after the protocol smoke checks pass.

Prepare HAM10000 once:

```bash
python scripts/prepare_ham10000.py \
  --metadata-csv /content/drive/MyDrive/HAM10000/HAM10000_metadata.csv \
  --image-dir /content/drive/MyDrive/HAM10000/images \
  --split-csv /content/drive/MyDrive/HAM10000/splits.csv \
  --report-json /content/drive/MyDrive/TrustQueryNet/artifacts/ham10000_dataset_report.json
```

Run one corrected Q1 smoke experiment:

```bash
python scripts/run_experiment.py --config configs/q1_ham10000_convnext_repair.yaml
```

For heavier Colab runs, especially multi-seed experiments, stage HAM10000 into local runtime storage first so training does not stream images from Google Drive:

```bash
python scripts/stage_ham10000_local.py \
  --source-root /content/drive/MyDrive/HAM10000 \
  --target-root /content/HAM10000-local
```

Then point the Colab config at:

- metadata: `/content/HAM10000-local/HAM10000_metadata.csv`
- images: `/content/HAM10000-local/images`
- splits: `/content/HAM10000-local/splits.csv`

## Export final artifacts

The export bundle script packages:

- `results_table.csv`
- `results_table.md`
- `summary.json`
- selected metrics, split manifests, configs, and plots from the chosen runs

Example:

```bash
python scripts/export_results_bundle.py \
  --runs-root artifacts/runs \
  --run q1-ham10000-convnext-repair \
  --run q1-ham10000-convnext-random-repair \
  --output-root artifacts/exports \
  --bundle-name trustquerynet-q1-rerun-bundle
```

## Paper and portfolio docs

Project-facing write-up assets live here:

- [docs/mini_paper_draft.md](docs/mini_paper_draft.md)
- [docs/portfolio_copy.md](docs/portfolio_copy.md)
- [docs/supplementary_appendix.md](docs/supplementary_appendix.md)

## Current boundaries

This repo slice is strong enough for a serious Q1-oriented upgrade path, but it does not yet claim:

- that the corrected rerun suite has already been executed
- formal paired significance testing throughout the full manuscript
- prospective or clinician-in-the-loop evaluation
- advanced methods such as SelectiveNet, DivideMix, or SWAG as verified final results

The implemented foundation is intentionally modular so the corrected publication reruns can proceed without rewriting the core pipeline.
