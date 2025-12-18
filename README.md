# TrustQueryNet

TrustQueryNet is a local-first, research-oriented image classification pipeline for noisy labels, uncertainty estimation, active querying, and selective prediction.

This repository is intentionally structured so the exact same codebase can be used:

- on a local Apple Silicon laptop for quick iterations and pilot runs
- on Colab or an A100-backed VM for heavier experiments

## What is in scope right now

This first cut implements a practical v1:

- reproducible dataset splits
- symmetric and transition-matrix label noise
- pretrained backbones through `timm`
- baseline and robust losses
- temperature scaling
- uncertainty-based acquisition scoring
- selective risk / coverage metrics
- a notebook orchestrator plus a CLI entrypoint

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
python scripts/run_experiment.py --config configs/quick_cifar100.yaml
```

## Current status

Verified locally:

- package installs in editable mode
- `pytest -q` passes
- `configs/quick_cifar100.yaml` runs end-to-end on Apple Silicon
- `configs/quick_cifar100_active.yaml` is available for active-learning smoke checks

Not yet verified end-to-end:

- real HAM10000 download and pilot run
- Colab execution against your own Drive-mounted HAM10000 copy

## Notebook

Open [notebooks/trustquerynet_local.ipynb](/Users/stelioszacharioudakis/Documents/TrustQueryNet/notebooks/trustquerynet_local.ipynb) for the local-first runner notebook.

For Colab handoff, use [notebooks/trustquerynet_colab.ipynb](/Users/stelioszacharioudakis/Documents/TrustQueryNet/notebooks/trustquerynet_colab.ipynb).

## Colab handoff

Once the local scaffold is validated, the same repo can be pushed to GitHub and run in Colab with:

```bash
git clone <YOUR_REPO_URL>
cd TrustQueryNet
pip install -q -r requirements-colab.txt
pip install -q -e . --no-deps
python scripts/run_experiment.py --config configs/pilot_ham10000.yaml
```

Before the first HAM10000 run, prepare the split/report artifacts once:

```bash
python scripts/prepare_ham10000.py \
  --metadata-csv data/ham10000/HAM10000_metadata.csv \
  --image-dir data/ham10000/images \
  --split-csv data/ham10000/splits.csv \
  --report-json artifacts/ham10000_dataset_report.json
```
