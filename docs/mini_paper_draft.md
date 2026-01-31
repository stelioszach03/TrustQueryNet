# TrustQueryNet: Trustworthy Skin Lesion Classification under Noisy Labels, Active Queries, and Selective Prediction

## Abstract

This project studies trustworthy dermatoscopic image classification when supervision is imperfect and annotation budget is limited. We build a modular HAM10000 pipeline that combines group-aware splitting, synthetic class-dependent label noise, active querying, calibration, and selective prediction. The final verified run uses a balanced ConvNeXt-Tiny configuration with weighted sampling, cross-entropy with label smoothing, transition-matrix noise, and entropy-based active querying. On the exported calibrated test set, the final model reaches `0.8330` accuracy, `0.7372` macro-F1, `0.9434` macro-AUROC, and `0.0250` expected calibration error. Relative to the pilot run, the full configuration improves macro-F1 by roughly `+0.242`, substantially strengthening minority-class performance while preserving good calibration.

## 1. Introduction

Many applied medical imaging projects report only headline accuracy, even though practical deployment depends on broader trustworthiness properties. In dermatoscopic lesion classification, label quality can be heterogeneous, classes are strongly imbalanced, and high-confidence mistakes can be costly. TrustQueryNet was built to address this gap with a single reproducible pipeline that supports:

- training under noisy labels
- uncertainty-aware querying of suspicious samples
- post-hoc calibration for better probability estimates
- selective prediction via risk-coverage analysis

The project is intentionally engineered as a research-grade portfolio artifact rather than a one-off notebook. Every major stage writes persistent artifacts so experiments can be rerun, compared, and packaged into a compact paper bundle.

## 2. Method

### 2.1 Data and split protocol

The primary dataset is HAM10000, a 7-class dermatoscopic benchmark with multiple images per lesion and strong class imbalance. To avoid leakage, the pipeline performs group-aware splitting at the lesion level rather than at the image level. Split manifests, dataset reports, and noise manifests are persisted to disk.

### 2.2 Noise model

Training labels are corrupted with a class-dependent transition matrix meant to simulate clinically plausible confusion between visually similar lesion types. The observed labels are stored separately from the clean labels, allowing repaired labels to be tracked through active querying.

### 2.3 Model and optimization

Two verified HAM10000 configurations were used:

- `pilot-ham10000`: EfficientNet-B0, generalized cross-entropy, entropy querying
- `full-ham10000-convnext-balanced`: ConvNeXt-Tiny, weighted sampling, cross-entropy with `0.05` label smoothing, entropy querying

The final configuration is important because earlier ConvNeXt runs collapsed to the majority `nv` class. Weighted sampling corrected the imbalance pressure and produced the strongest overall result.

### 2.4 Trustworthiness components

TrustQueryNet evaluates more than classification accuracy. It includes:

- temperature scaling for calibration
- expected calibration error and Brier score
- selective risk and coverage metrics
- reliability and risk-coverage plots

These components make the project suitable for a trustworthy ML narrative instead of a pure accuracy narrative.

## 3. Experimental setup

The final paper bundle compares two exported HAM10000 runs:

| Run | Backbone | Main training recipe | Querying | Best use |
| --- | --- | --- | --- | --- |
| `pilot-ham10000` | EfficientNet-B0 | GCE under transition-matrix noise | Entropy, 2 rounds | fast sanity-checked baseline |
| `full-ham10000-convnext-balanced` | ConvNeXt-Tiny | weighted sampling + CE + label smoothing | Entropy, 2 rounds | main result |

Both runs use persistent split manifests and calibrated test evaluation. The full run is the one to feature as the headline result.

## 4. Results

### 4.1 Main quantitative comparison

Calibrated test metrics from the exported bundle:

| Run | Accuracy | Macro-F1 | ECE | Macro-AUROC | Coverage@0.5 | Risk@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pilot-ham10000` | 0.7785 | 0.4950 | 0.0599 | 0.8827 | 0.8175 | 0.1499 |
| `full-ham10000-convnext-balanced` | 0.8330 | 0.7372 | 0.0250 | 0.9434 | 0.9441 | 0.1427 |

### 4.2 Interpretation

The full configuration materially improves the project along all of the metrics that matter for a trustworthy classifier:

- accuracy improves by roughly `+0.0545`
- macro-F1 improves by roughly `+0.2422`
- macro-AUROC improves by roughly `+0.0607`
- ECE drops from `0.0599` to `0.0250`

The macro-F1 gain is the most important result because it suggests the model is not merely optimizing the dominant nevus class. The low final ECE also supports the project’s calibration and abstention story.

### 4.3 Figures to use

From the exported bundle, the most useful paper figures are:

- `runs/full-ham10000-convnext-balanced/active_round_2/plots/reliability_calibrated.png`
- `runs/full-ham10000-convnext-balanced/active_round_2/plots/risk_coverage_calibrated.png`
- `runs/pilot-ham10000/active_round_2/plots/reliability_calibrated.png`
- `runs/pilot-ham10000/active_round_2/plots/risk_coverage_calibrated.png`

These support both the main result and the pilot-to-full improvement story.

## 5. Limitations

This project is strong as a reproducible mini-paper and portfolio artifact, but the current evidence should still be framed honestly:

- results are currently based on single-seed verified runs
- evaluation is limited to HAM10000
- advanced methods such as SelectiveNet, DivideMix, SWAG, and multi-seed significance testing are not yet part of the verified final package

These are reasonable next steps, but they are not required for the current project to be compelling.

## 6. Conclusion

TrustQueryNet demonstrates that a trustworthy ML framing can be implemented end-to-end in a clean, modular research pipeline. The final balanced ConvNeXt-Tiny run achieves strong calibrated performance on HAM10000 while preserving the broader narrative around noisy supervision, selective querying, and calibrated decision-making. For a portfolio project or master’s application, the current result is already strong enough to present as a polished, research-oriented systems project.
