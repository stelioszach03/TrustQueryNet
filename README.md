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

Recommended workflow:

- use [notebooks/trustquerynet_local.ipynb](/Users/stelioszacharioudakis/Documents/TrustQueryNet/notebooks/trustquerynet_local.ipynb) for local-first exploration
- use [notebooks/trustquerynet_colab.ipynb](/Users/stelioszacharioudakis/Documents/TrustQueryNet/notebooks/trustquerynet_colab.ipynb) in browser Colab UI for Drive-backed runs and artifact persistence

## Final HAM10000 results

The current main result is the balanced ConvNeXt-Tiny run:

- config: [configs/full_ham10000_convnext.yaml](/Users/stelioszacharioudakis/Documents/TrustQueryNet/configs/full_ham10000_convnext.yaml)
- exported run name: `full-ham10000-convnext-balanced`
- setup: ConvNeXt-Tiny, weighted sampler, cross-entropy with label smoothing, transition-matrix noise, entropy-based active querying

Calibrated test metrics from the exported paper bundle:

| Run | Accuracy | Macro-F1 | ECE | Macro-AUROC | Coverage@0.5 | Risk@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pilot-ham10000` | 0.7785 | 0.4950 | 0.0599 | 0.8827 | 0.8175 | 0.1499 |
| `full-ham10000-convnext-balanced` | 0.8330 | 0.7372 | 0.0250 | 0.9434 | 0.9441 | 0.1427 |

Why this matters:

- the full run improves macro-F1 by roughly `+0.242`
- calibration remains strong with `ECE ~= 0.025`
- the exported plots show a clear risk-coverage and reliability story for a trustworthy ML write-up

These are single-seed project results, not a multi-seed benchmark claim.

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

Run the current best full experiment:

```bash
python scripts/run_experiment.py --config configs/colab_full_ham10000_convnext.yaml
```

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
  --run full-ham10000-convnext-balanced \
  --output-root /content/drive/MyDrive/TrustQueryNet/exports \
  --bundle-name trustquerynet-paper-bundle
```

## Paper and portfolio docs

Project-facing write-up assets live here:

- [docs/mini_paper_draft.md](/Users/stelioszacharioudakis/Documents/TrustQueryNet/docs/mini_paper_draft.md)
- [docs/portfolio_copy.md](/Users/stelioszacharioudakis/Documents/TrustQueryNet/docs/portfolio_copy.md)

## Current boundaries

This repo slice is strong enough for a portfolio project and mini-paper, but it does not yet claim:

- multi-seed statistical stability studies
- external dataset validation
- advanced methods such as SelectiveNet, DivideMix, or SWAG as verified final results

The implemented foundation is intentionally modular so those can be added later without rewriting the core pipeline.
