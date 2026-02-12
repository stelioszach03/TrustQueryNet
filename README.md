# TrustQueryNet

TrustQueryNet is a local-first, research-oriented image classification pipeline for noisy labels, uncertainty estimation, active querying, and selective prediction.

The current project slice targets dermatoscopic skin lesion classification on HAM10000 and is designed to run:

- locally on Apple Silicon for quick iteration
- in Colab with GPU for pilot and full experiments

## Implemented scope

This repository currently includes:

- reproducible group-aware HAM10000 splits
- symmetric and transition-matrix label noise
- pretrained backbones through `timm`
- cross-entropy, generalized cross-entropy, and symmetric cross-entropy losses
- temperature scaling for calibration
- uncertainty-aware active querying
- selective risk / coverage evaluation
- Colab export tooling for final paper bundles

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

- use [notebooks/trustquerynet_local.ipynb](/Users/stelioszacharioudakis/Documents/TrustQueryNet/notebooks/trustquerynet_local.ipynb) for local-first exploration
- use [notebooks/trustquerynet_colab.ipynb](/Users/stelioszacharioudakis/Documents/TrustQueryNet/notebooks/trustquerynet_colab.ipynb) in browser Colab UI for Drive-backed runs and artifact persistence

## Current paper-facing results

The final paper-facing model is now the `no-weighted-sampler` ConvNeXt-Tiny variant, because the ablation study showed it outperformed the earlier weighted-sampler setup under multi-seed evaluation.

Main internal result:

- config: [configs/full_ham10000_convnext_no_weighted.yaml](/Users/stelioszacharioudakis/Documents/TrustQueryNet/configs/full_ham10000_convnext_no_weighted.yaml)
- exported multi-seed run: `full-ham10000-convnext-no-weighted-multiseed`
- setup: ConvNeXt-Tiny, cross-entropy with `0.05` label smoothing, transition-matrix noise, entropy-based active querying, lesion-level fixed split, `5` seeds

Calibrated internal HAM10000 metrics (`mean ± std` across seeds):

| Setting | Accuracy | Macro-F1 | ECE | Macro-AUROC | Coverage@0.5 | Risk@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Full model (no weighted sampler)` | 0.8454 ± 0.0054 | 0.7265 ± 0.0080 | 0.0375 ± 0.0105 | 0.9599 ± 0.0048 | 0.9467 ± 0.0039 | 0.1303 ± 0.0077 |
| `Full model (weighted sampler)` | 0.8074 ± 0.0095 | 0.6933 ± 0.0244 | 0.0327 ± 0.0053 | 0.9410 ± 0.0096 | 0.9313 ± 0.0095 | 0.1630 ± 0.0101 |
| `No query rounds` | 0.7989 ± 0.0072 | 0.6742 ± 0.0234 | 0.0380 ± 0.0114 | 0.9327 ± 0.0065 | 0.9250 ± 0.0062 | 0.1668 ± 0.0093 |

External validation on the official ISIC 2019 test set for the main internal model:

| External setting | Accuracy | Macro-F1 | ECE | Macro-AUROC | Coverage@0.5 | Risk@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ISIC 2019 external test` | 0.5551 ± 0.0263 | 0.4256 ± 0.0312 | 0.2236 ± 0.0462 | 0.8237 ± 0.0158 | 0.8576 ± 0.0315 | 0.3966 ± 0.0357 |

Why this matters:

- active querying helps relative to the `no-query-rounds` ablation
- removing the weighted sampler improves both internal discrimination and selective risk in the current setup
- external validation reveals a substantial domain-shift gap, which strengthens the paper’s trustworthiness narrative instead of hiding it
- internal temperature scaling yields only a small internal ECE improvement and does not transfer well under external shift

## Multi-seed benchmarking

For Q1-style reporting, run the current best config across multiple seeds and aggregate the calibrated test metrics:

```bash
python scripts/run_multiseed_experiment.py \
  --config configs/full_ham10000_convnext_no_weighted.yaml \
  --seeds 42 52 62 72 82 \
  --resume-existing
```

This writes a sibling directory next to the base run output, for example:

- `artifacts/runs/full-ham10000-convnext-no-weighted-multiseed/seed-42`
- `artifacts/runs/full-ham10000-convnext-no-weighted-multiseed/aggregate_results.json`
- `artifacts/runs/full-ham10000-convnext-no-weighted-multiseed/aggregate_results.md`

On HAM10000, keep `split_csv` populated so the lesion-level split stays fixed across seeds while the training randomness changes.

## Ablation study workflow

Two focused ablation configs are included for the current ConvNeXt-Tiny HAM10000 setup:

- [configs/full_ham10000_convnext_no_weighted.yaml](/Users/stelioszacharioudakis/Documents/TrustQueryNet/configs/full_ham10000_convnext_no_weighted.yaml)
- [configs/full_ham10000_convnext_no_querying.yaml](/Users/stelioszacharioudakis/Documents/TrustQueryNet/configs/full_ham10000_convnext_no_querying.yaml)

Recommended Q1-style ablation set:

- `full-ham10000-convnext-no-weighted`: current main result
- `full-ham10000-convnext-balanced`: weighted-sampler comparison
- `full-ham10000-convnext-no-querying`: keeps the initial trusted fraction but disables query rounds

After running each one with the multi-seed runner, export a single paper-ready table:

```bash
python scripts/export_ablation_table.py \
  --runs-root /content/drive/MyDrive/TrustQueryNet/artifacts/runs \
  --run-spec full-ham10000-convnext-no-weighted-multiseed::Full\ model \
  --run-spec full-ham10000-convnext-balanced-multiseed::Weighted\ sampler \
  --run-spec full-ham10000-convnext-no-querying-multiseed::No\ query\ rounds \
  --output-dir /content/drive/MyDrive/TrustQueryNet/ablations/ham10000
```

## External validation workflow

The current external-validation slice targets the official `ISIC 2019` test set and maps its challenge labels into the HAM10000-style 7-class taxonomy:

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
  --multiseed-run-dir /content/drive/MyDrive/TrustQueryNet/artifacts/runs/full-ham10000-convnext-no-weighted-multiseed \
  --ground-truth-csv /content/drive/MyDrive/ISIC2019_external_test/ISIC_2019_Test_GroundTruth.csv \
  --metadata-csv /content/drive/MyDrive/ISIC2019_external_test/ISIC_2019_Test_Metadata.csv \
  --image-dir /content/drive/MyDrive/ISIC2019_external_test/images \
  --num-workers 0
```

This reuses the existing multi-seed checkpoints and writes an external validation summary with per-seed metrics, aggregate mean/std tables, and reliability / risk-coverage plots.

## Colab workflow

1. Open [notebooks/trustquerynet_colab.ipynb](/Users/stelioszacharioudakis/Documents/TrustQueryNet/notebooks/trustquerynet_colab.ipynb) in Colab UI.
2. Mount Drive in the notebook.
3. Pull the latest repo state.
4. Run the pilot or full experiment config.
5. Export a paper bundle with the script below.

Prepare HAM10000 once:

```bash
python scripts/prepare_ham10000.py \
  --metadata-csv /content/drive/MyDrive/HAM10000/HAM10000_metadata.csv \
  --image-dir /content/drive/MyDrive/HAM10000/images \
  --split-csv /content/drive/MyDrive/HAM10000/splits.csv \
  --report-json /content/drive/MyDrive/TrustQueryNet/artifacts/ham10000_dataset_report.json
```

Run the current best single-seed full experiment:

```bash
python scripts/run_experiment.py --config configs/colab_full_ham10000_convnext_no_weighted.yaml
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

The paper/export bundle script packages:

- `results_table.csv`
- `results_table.md`
- `summary.json`
- selected metrics, split manifests, configs, and plots from the chosen runs

Example Colab command:

```bash
python scripts/export_results_bundle.py \
  --runs-root /content/drive/MyDrive/TrustQueryNet/artifacts/runs \
  --run pilot-ham10000 \
  --run full-ham10000-convnext-no-weighted \
  --output-root /content/drive/MyDrive/TrustQueryNet/exports \
  --bundle-name trustquerynet-paper-bundle
```

## Paper and portfolio docs

Project-facing write-up assets live here:

- [docs/mini_paper_draft.md](/Users/stelioszacharioudakis/Documents/TrustQueryNet/docs/mini_paper_draft.md)
- [docs/portfolio_copy.md](/Users/stelioszacharioudakis/Documents/TrustQueryNet/docs/portfolio_copy.md)
- [docs/supplementary_appendix.md](/Users/stelioszacharioudakis/Documents/TrustQueryNet/docs/supplementary_appendix.md)

## Current boundaries

This repo slice is strong enough for a portfolio project and mini-paper, but it does not yet claim:

- formal paired significance testing throughout the full manuscript
- prospective or clinician-in-the-loop evaluation
- advanced methods such as SelectiveNet, DivideMix, or SWAG as verified final results

The implemented foundation is intentionally modular so those can be added later without rewriting the core pipeline.
