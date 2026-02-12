# TrustQueryNet: Trustworthy Skin Lesion Classification under Noisy Labels, Active Queries, and External Validation

## Abstract

This project studies trustworthy dermatoscopic image classification under noisy supervision and limited trusted annotation. We build a modular pipeline around HAM10000 that combines lesion-level group-aware splitting, class-dependent label noise, uncertainty-aware active querying, post-hoc calibration, and selective prediction analysis. The final paper-facing model is a ConvNeXt-Tiny configuration trained without weighted sampling, using cross-entropy with label smoothing and entropy-based query rounds. Across `5` seeds on internal HAM10000 evaluation, it achieves `0.8454 ± 0.0054` calibrated accuracy, `0.7265 ± 0.0080` macro-F1, `0.9599 ± 0.0048` macro-AUROC, and `0.0375 ± 0.0105` expected calibration error. An ablation study shows that active querying improves performance over a no-query baseline, while weighted sampling underperforms in the current noisy-label setting. External validation on the official ISIC 2019 test set reveals a substantial generalization drop to `0.5551 ± 0.0263` calibrated accuracy and `0.4256 ± 0.0312` macro-F1, highlighting distribution shift as the main remaining challenge.

## 1. Introduction

Medical image classification projects often report only in-distribution accuracy, even though real use depends on label quality, confidence quality, and robustness under shift. Dermatoscopic lesion classification is a good example: classes are heavily imbalanced, labels may come from mixed-quality supervision pipelines, and model confidence matters because high-confidence errors can be clinically costly.

TrustQueryNet was built to make those concerns first-class instead of post-hoc:

- noisy-label learning is explicit rather than ignored
- trusted supervision is treated as limited and budgeted
- uncertainty and calibration are evaluated directly
- selective prediction is measured through risk-coverage analysis
- all stages write persistent artifacts for reproducibility

The resulting codebase is both a research artifact and an engineering artifact: local development, Colab runs, export bundles, and paper tables all use the same package and config stack.

## 2. Method

### 2.1 Data and split protocol

The primary development dataset is HAM10000, a `7`-class dermatoscopic benchmark with multiple images per lesion and severe class imbalance. To prevent leakage, TrustQueryNet performs lesion-level group-aware splitting and persists the split manifest to disk. Clean labels, observed labels, and query/trust state are tracked separately inside the dataset manifest.

### 2.2 Noise model

Training labels are corrupted with a class-dependent transition matrix designed to reflect plausible visual confusion among lesion categories. This gives the pipeline a reproducible noisy-label setting while preserving the original clean labels for evaluation and simulated repair.

### 2.3 Model family and optimization

The verified HAM10000 study uses:

- EfficientNet-B0 as a fast pilot baseline
- ConvNeXt-Tiny as the full model family
- cross-entropy or generalized cross-entropy losses depending on the slice
- optional weighted sampling for imbalance control
- temperature scaling for post-hoc calibration

The final internal model is:

- `full-ham10000-convnext-no-weighted`
- ConvNeXt-Tiny
- cross-entropy with `0.05` label smoothing
- transition-matrix noise
- entropy-based active querying
- `5` fixed-split seeds

### 2.4 Trustworthiness components

TrustQueryNet evaluates:

- macro-F1 and macro-AUROC for class-balanced discrimination
- expected calibration error (ECE) and reliability diagrams
- coverage and selective risk at thresholded confidence levels
- failure behavior under external distribution shift

These components frame the project as trustworthy ML rather than a pure benchmark-optimization exercise.

## 3. Experimental setup

### 3.1 Internal evaluation

All internal HAM10000 experiments use the same lesion-level split logic and run under a fixed transition-matrix noise configuration. The paper-facing results are reported across `5` seeds with `mean ± std`.

### 3.2 Ablation study

The ablation study compares three focused variants:

| Variant | Difference from baseline | Purpose |
| --- | --- | --- |
| `Full model` | ConvNeXt-Tiny, no weighted sampler, querying enabled | main internal result |
| `Weighted sampler` | same setup but with weighted sampling | isolate imbalance-handling effect |
| `No query rounds` | same setup but no repair rounds | isolate active querying effect |

### 3.3 External validation

External validation is performed on the official ISIC 2019 test set. To match the HAM10000-style `7`-class taxonomy, ISIC 2019 labels are mapped as follows:

- `MEL -> mel`
- `NV -> nv`
- `BCC -> bcc`
- `AK -> akiec`
- `SCC -> akiec`
- `BKL -> bkl`
- `DF -> df`
- `VASC -> vasc`
- `UNK` is filtered

This produces an external evaluation set that is clinically related but distributionally distinct from HAM10000.

## 4. Results

### 4.1 Internal multi-seed performance

The final internal result uses the no-weighted-sampler ConvNeXt-Tiny model:

| Setting | Accuracy | Macro-F1 | ECE | Macro-AUROC | Coverage@0.5 | Risk@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Full model (no weighted sampler)` | `0.8454 ± 0.0054` | `0.7265 ± 0.0080` | `0.0375 ± 0.0105` | `0.9599 ± 0.0048` | `0.9467 ± 0.0039` | `0.1303 ± 0.0077` |

This is the main quantitative result to headline in the manuscript. The earlier single-seed `0.8330 / 0.7372 / 0.9434` run was useful during model development, but the multi-seed result is the paper-grade claim.

### 4.2 Ablation study

| Variant | Accuracy | Macro-F1 | ECE | Macro-AUROC | Coverage@0.5 | Risk@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Full model` | `0.8454 ± 0.0054` | `0.7265 ± 0.0080` | `0.0375 ± 0.0105` | `0.9599 ± 0.0048` | `0.9467 ± 0.0039` | `0.1303 ± 0.0077` |
| `Weighted sampler` | `0.8074 ± 0.0095` | `0.6933 ± 0.0244` | `0.0327 ± 0.0053` | `0.9410 ± 0.0096` | `0.9313 ± 0.0095` | `0.1630 ± 0.0101` |
| `No query rounds` | `0.7989 ± 0.0072` | `0.6742 ± 0.0234` | `0.0380 ± 0.0114` | `0.9327 ± 0.0065` | `0.9250 ± 0.0062` | `0.1668 ± 0.0093` |

Two conclusions are especially important:

- active querying improves performance relative to the no-query baseline
- weighted sampling does not help in the current noisy-label setting and is outperformed by the no-weighted baseline

This second result materially changed the paper story: the best-performing model is not the earlier weighted-sampler recipe but the cleaner no-weighted configuration.

### 4.3 External validation

The same paper-facing internal model was evaluated on the official ISIC 2019 test set:

| External setting | Accuracy | Macro-F1 | ECE | Macro-AUROC | Coverage@0.5 | Risk@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ISIC 2019 external test` | `0.5551 ± 0.0263` | `0.4256 ± 0.0312` | `0.2236 ± 0.0462` | `0.8237 ± 0.0158` | `0.8576 ± 0.0315` | `0.3966 ± 0.0357` |

Compared with internal evaluation, this is a substantial drop:

- accuracy falls from `0.8454` to `0.5551`
- macro-F1 falls from `0.7265` to `0.4256`
- macro-AUROC falls from `0.9599` to `0.8237`

This is the clearest evidence of distribution shift in the current study, and it should be treated as a central finding rather than as an inconvenient side result.

### 4.4 Calibration behavior

Internal temperature scaling produces only a small ECE gain for the main model:

- internal uncalibrated ECE: `0.0387 ± 0.0070`
- internal calibrated ECE: `0.0375 ± 0.0105`

On the external ISIC 2019 test set, temperature scaling fitted on the internal validation set does not transfer well:

- external uncalibrated ECE: `0.2011 ± 0.0372`
- external calibrated ECE: `0.2236 ± 0.0462`

This is an important trustworthy-ML result in its own right: calibration fitted in-distribution should not be assumed to remain reliable under external shift.

## 5. Discussion

The project now supports three defensible claims.

First, the internal HAM10000 pipeline is strong and stable. The main ConvNeXt-Tiny configuration reaches `0.7265 ± 0.0080` macro-F1 and `0.9599 ± 0.0048` macro-AUROC across `5` seeds, which is strong enough to support a serious research-style write-up rather than a toy demo.

Second, the ablation results are informative rather than cosmetic. Active querying contributes measurable gains, while weighted sampling is not universally beneficial under the present noisy-label recipe. That makes the paper more credible, because the study reports what actually helped rather than what was expected to help.

Third, the external validation result meaningfully strengthens the manuscript. The drop on ISIC 2019 shows that the model is not simply “solved” and that trustworthiness claims should be made with explicit awareness of domain shift. The paper is stronger because it contains that result.

## 6. Limitations

The current package should still be framed honestly:

- external validation is limited to one external dataset
- the external class mapping collapses `SCC` into `akiec` and filters `UNK`
- calibration transfer under shift is weak
- the manuscript does not yet include formal paired significance testing throughout
- there is no prospective clinical evaluation or reader study
- advanced extensions such as SelectiveNet, DivideMix, or SWAG are outside the verified final slice

These are appropriate limitations for a strong student-led Q1-oriented applied ML submission.

## 7. Conclusion

TrustQueryNet now supports a publication-style narrative that is both strong and honest. Internally, the no-weighted ConvNeXt-Tiny model delivers stable multi-seed performance on HAM10000. Ablations show that active querying helps and that weighted sampling is not the best choice in the current noisy-label setting. External validation on ISIC 2019 reveals a clear generalization gap and weak calibration transfer under shift. Taken together, these results make the project substantially stronger as a trustworthy ML paper than a single high internal accuracy number would.

## 8. Figures and tables to include

Recommended main-paper assets:

- internal ablation table from `ablations/ham10000/ablation_table.md`
- reliability diagram and risk-coverage plot for the main internal model
- external reliability diagram and risk-coverage plot from the ISIC 2019 evaluation
- one short internal-vs-external summary table

Recommended supplementary assets:

- seed-level internal results table
- seed-level external validation table
- dataset mapping note for ISIC 2019 labels
- exact reproduction commands and config paths
