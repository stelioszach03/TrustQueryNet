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

Repository map:

- [configs/README.md](configs/README.md) explains which config families are final paper-facing versus preserved development provenance
- [artifacts/README.md](artifacts/README.md) explains which lightweight artifact summaries are intentionally tracked in git
- [paper/README.md](paper/README.md) describes the manuscript sources and paper rebuild workflow

## Current verified results

The numbers below are the **current corrected results** from the locked `e12` ConvNeXt-Tiny publication recipe. They are the main evidence slice for the paper-facing trustworthy-ML study in this repo.

Primary internal comparison on HAM10000 (`mean ± std` across seeds):

| Setting | Accuracy | Macro-F1 | ECE | Macro-AUROC | AURC | Coverage@0.5 | Risk@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Repair` | 0.8350 ± 0.0059 | 0.7152 ± 0.0216 | 0.0445 ± 0.0090 | 0.9382 ± 0.0133 | 0.0728 ± 0.0186 | 0.9525 ± 0.0074 | 0.1428 ± 0.0054 |
| `No repair` | 0.8321 ± 0.0071 | 0.7145 ± 0.0130 | 0.0356 ± 0.0075 | 0.9460 ± 0.0128 | 0.0622 ± 0.0093 | 0.9480 ± 0.0066 | 0.1449 ± 0.0078 |
| `Random repair` | 0.8319 ± 0.0051 | 0.7105 ± 0.0239 | 0.0356 ± 0.0071 | 0.9521 ± 0.0128 | 0.0588 ± 0.0145 | 0.9376 ± 0.0100 | 0.1402 ± 0.0107 |

Noisy-label anchors:

| Setting | Accuracy | Macro-F1 | ECE | Macro-AUROC | AURC | Coverage@0.5 | Risk@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Clean upper` | 0.8521 ± 0.0055 | 0.7330 ± 0.0288 | 0.0389 ± 0.0176 | 0.9583 ± 0.0076 | 0.0502 ± 0.0095 | 0.9618 ± 0.0107 | 0.1306 ± 0.0089 |
| `GCE no repair` | 0.6774 ± 0.0012 | 0.1224 ± 0.0122 | 0.0268 ± 0.0381 | 0.5782 ± 0.1279 | 0.2399 ± 0.1059 | 0.9639 ± 0.0626 | 0.3044 ± 0.0327 |

External validation on the official ISIC 2019 test set:

| Setting | Accuracy | Macro-F1 | ECE | Macro-AUROC | AURC | Coverage@0.5 | Risk@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Repair external` | 0.5692 ± 0.0145 | 0.4427 ± 0.0117 | 0.2000 ± 0.0253 | 0.8125 ± 0.0180 | 0.2804 ± 0.0299 | 0.8526 ± 0.0230 | 0.3805 ± 0.0228 |
| `No repair external` | 0.5630 ± 0.0078 | 0.4288 ± 0.0123 | 0.2016 ± 0.0123 | 0.8168 ± 0.0154 | 0.2749 ± 0.0183 | 0.8498 ± 0.0264 | 0.3838 ± 0.0073 |
| `Random repair external` | 0.5591 ± 0.0203 | 0.4311 ± 0.0328 | 0.2149 ± 0.0395 | 0.8114 ± 0.0337 | 0.2905 ± 0.0669 | 0.8654 ± 0.0383 | 0.3964 ± 0.0269 |

Why these corrected results matter:

- budgeted trusted-label repair delivers only modest point-performance gains over strong baselines
- no-repair and random-repair remain highly competitive across calibration and selective metrics
- the clean-label upper bound shows remaining headroom under noisy supervision
- generalized cross-entropy collapses under the chosen corruption regime
- external validation reveals a substantial domain-shift gap even after internal calibration
- an overlap audit found `0` exact duplicate images between HAM10000 and the mapped ISIC 2019 external slice
- paired seed-level comparisons on calibrated macro-F1 keep the main claims intentionally modest rather than threshold-driven

## Q1 rerun config family

The corrected rerun family for the Q1-oriented paper path lives in:

- [configs/q1_ham10000_convnext_repair.yaml](configs/q1_ham10000_convnext_repair.yaml)
- [configs/q1_ham10000_convnext_no_repair.yaml](configs/q1_ham10000_convnext_no_repair.yaml)
- [configs/q1_ham10000_convnext_random_repair.yaml](configs/q1_ham10000_convnext_random_repair.yaml)
- [configs/q1_ham10000_convnext_clean_upper.yaml](configs/q1_ham10000_convnext_clean_upper.yaml)
- [configs/q1_ham10000_convnext_gce_no_repair.yaml](configs/q1_ham10000_convnext_gce_no_repair.yaml)
- [configs/q1_ham10000_convnext_weighted_secondary.yaml](configs/q1_ham10000_convnext_weighted_secondary.yaml)

These configs support:

- explicit checkpoint-policy selection
- AMP-enabled training
- longer training with warmup and early stopping hooks
- dense selective metrics / AURC defaults
- deconfounded no-repair baseline
- matched-budget random-repair baseline
- clean-label upper bound
- robust-loss baseline comparison

The corrected `e12` publication recipe has already been executed for the main internal and external comparisons. Keep the earlier exploratory runs frozen as separate development artifacts.

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
  --runs-root artifacts/final_evidence \
  --run q1-ham10000-convnext-repair-final-e12-multiseed \
  --run q1-ham10000-convnext-no-repair-final-e12-multiseed \
  --run q1-ham10000-convnext-random-repair-final-e12-multiseed \
  --run q1-ham10000-convnext-repair-final-e12-multiseed-external-isic2019-test \
  --output-root artifacts/exports \
  --bundle-name trustquerynet-q1-paper-bundle
```

To export the paper-facing paired significance table from the locked final evidence slice:

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

## Paper and portfolio docs

Project-facing write-up assets live here:

- [docs/mini_paper_draft.md](docs/mini_paper_draft.md)
- [docs/paper_abstract.md](docs/paper_abstract.md)
- [docs/final_results_tables.md](docs/final_results_tables.md)
- [docs/paper_package_manifest.md](docs/paper_package_manifest.md)
- [docs/portfolio_copy.md](docs/portfolio_copy.md)
- [docs/supplementary_appendix.md](docs/supplementary_appendix.md)

Paper-facing LaTeX sources and compiled PDFs live here:

- [paper/README.md](paper/README.md)
- [paper/trustquerynet_abstract.pdf](paper/trustquerynet_abstract.pdf)
- [paper/trustquerynet_paper.pdf](paper/trustquerynet_paper.pdf)
- [paper/references.bib](paper/references.bib)

Lightweight final evidence summaries that support the public paper package live here:

- [artifacts/final_evidence](artifacts/final_evidence)
- [artifacts/paper_tables](artifacts/paper_tables)
- [artifacts/overlap/ham10000-isic2019/overlap_report.json](artifacts/overlap/ham10000-isic2019/overlap_report.json)

## Current boundaries

This repo now contains the corrected core evidence suite, but it does not yet claim:

- exhaustive per-metric or per-class significance testing beyond the prespecified main comparisons
- clinician-in-the-loop or prospective evaluation
- manual adjudication of every perceptual-hash overlap candidate
- advanced methods such as SelectiveNet, DivideMix, or SWAG as verified final results

The implemented foundation is intentionally modular so additional reporting or manuscript polish can proceed without rewriting the core pipeline, but the current paper claim should stay centered on **budgeted trusted-label repair under simulated oracle supervision** and its limited external trustworthiness gains.
