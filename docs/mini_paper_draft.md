# TrustQueryNet: Working Manuscript Draft for Budgeted Trusted-Label Repair under Simulated Label Corruption and External Shift

> This draft is now a manuscript scaffold for the corrected Q1 rerun suite. The quantitative results quoted here are the **currently verified exploratory results**, not the final post-fix publication claims.

## Abstract

This project studies trustworthy dermatoscopic image classification under noisy supervision and limited trusted annotation. We build a modular pipeline around HAM10000 that combines lesion-level group-aware splitting, class-dependent label noise, **budgeted trusted-label repair under simulated oracle supervision**, post-hoc calibration, and selective prediction analysis. The current verified exploratory model is a ConvNeXt-Tiny configuration trained without weighted sampling, using cross-entropy with label smoothing and entropy-based repair rounds. Across `5` seeds on internal HAM10000 evaluation, it achieves `0.8454 ± 0.0054` calibrated accuracy, `0.7265 ± 0.0080` macro-F1, `0.9599 ± 0.0048` macro-AUROC, and `0.0375 ± 0.0105` expected calibration error. External validation on the official ISIC 2019 test set reveals a substantial generalization drop to `0.5551 ± 0.0263` calibrated accuracy and `0.4256 ± 0.0312` macro-F1, highlighting distribution shift as the main remaining challenge. The corrected Q1 rerun suite will replace the exploratory comparisons with a deconfounded no-repair baseline, a matched-budget random-repair baseline, a clean-label upper bound, and a robust-loss baseline.

## 1. Introduction

Medical image classification work often reports only in-distribution discrimination, even though real use depends on label quality, confidence quality, and robustness under shift. Dermatoscopic lesion classification is a good example: classes are heavily imbalanced, labels may come from mixed-quality supervision pipelines, and model confidence matters because high-confidence errors can be clinically costly.

TrustQueryNet was built to make those concerns first-class rather than post-hoc:

- noisy-label learning is explicit rather than ignored
- trusted supervision is treated as limited and budgeted
- repair of noisy labels is simulated explicitly
- calibration and selective prediction are evaluated directly
- all stages write persistent artifacts for reproducibility

The resulting codebase is both a research artifact and an engineering artifact: local development, Colab runs, export bundles, and paper tables all use the same package and config stack.

## 2. Method

### 2.1 Data and split protocol

The primary development dataset is HAM10000, a `7`-class dermatoscopic benchmark with multiple images per lesion and severe class imbalance. To prevent leakage, TrustQueryNet performs lesion-level group-aware splitting and persists the split manifest to disk. Clean labels, observed labels, and trust/repair state are tracked separately inside the dataset manifest.

### 2.2 Noise model

Training labels are corrupted with a class-dependent transition matrix designed to reflect plausible visual confusion among lesion categories. This gives the pipeline a reproducible noisy-label setting while preserving the original clean labels for evaluation and simulated correction.

### 2.3 Model family and optimization

The current verified HAM10000 slice uses:

- EfficientNet-B0 as a fast pilot baseline
- ConvNeXt-Tiny as the main model family
- cross-entropy, generalized cross-entropy, and symmetric cross-entropy implementations
- optional weighted sampling for imbalance control
- temperature scaling for post-hoc calibration

The exploratory internal model that currently anchors the repo is:

- `full-ham10000-convnext-no-weighted`
- ConvNeXt-Tiny
- cross-entropy with `0.05` label smoothing
- transition-matrix noise
- entropy-based trusted-label repair
- `5` fixed-split seeds

### 2.4 Budgeted trusted-label repair

The current repair protocol should be described precisely. After label corruption, a fraction of the training set is marked as initially trusted. The model is trained on the resulting noisy/trusted mix, uncertainty scores are computed on the unlabeled pool, and a fixed budget of selected samples is repaired by replacing `y_observed` with `y_clean`. This is a **simulated oracle supervision** protocol rather than a prospective active-learning deployment study.

### 2.5 Trustworthiness components

TrustQueryNet evaluates:

- macro-F1 and macro-AUROC for class-balanced discrimination
- expected calibration error (ECE) and reliability diagrams
- dense risk-coverage summaries and AURC
- failure behavior under external distribution shift

These components frame the project as trustworthy ML rather than a pure benchmark-optimization exercise.

## 3. Experimental protocol

### 3.1 Current exploratory internal evaluation

All internal HAM10000 experiments use the same lesion-level split logic and a fixed transition-matrix noise configuration. The current exploratory results are reported across `5` seeds with `mean ± std`.

### 3.2 Corrected Q1 rerun protocol

The corrected rerun suite should replace the exploratory comparisons with:

| Variant | Purpose |
| --- | --- |
| `Trusted repair` | main corrected result |
| `Random repair` | matched-budget repair baseline |
| `No repair` | zero-budget repair baseline |
| `Clean-label upper bound` | anchor for noisy supervision |
| `GCE no-repair` | robust-loss baseline |

Weighted sampling should remain a secondary ablation unless corrected reruns show that it is central.

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

This produces an external evaluation set that is clinically related but distributionally distinct from HAM10000. Before submission, this external slice must also be screened for exact and near-duplicate overlap against HAM10000.

## 4. Current exploratory results

### 4.1 Internal multi-seed performance

The current exploratory internal result uses the no-weighted-sampler ConvNeXt-Tiny model:

| Setting | Accuracy | Macro-F1 | ECE | Macro-AUROC | Coverage@0.5 | Risk@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Exploratory full model (no weighted sampler)` | `0.8454 ± 0.0054` | `0.7265 ± 0.0080` | `0.0375 ± 0.0105` | `0.9599 ± 0.0048` | `0.9467 ± 0.0039` | `0.1303 ± 0.0077` |

This is the strongest verified exploratory internal result, but it should not yet be presented as the final publication headline until the corrected rerun suite is complete.

### 4.2 Exploratory ablation signals

The current exploratory comparisons suggest two useful directions:

- repair helps relative to the exploratory no-query comparison
- weighted sampling does not help in the current noisy-label setting

However, the corrected paper must replace the exploratory no-query comparison with a deconfounded no-repair baseline and a matched-budget random-repair baseline before making a publication-grade repair claim.

### 4.3 External validation

The same exploratory internal model was evaluated on the official ISIC 2019 test set:

| External setting | Accuracy | Macro-F1 | ECE | Macro-AUROC | Coverage@0.5 | Risk@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ISIC 2019 external test` | `0.5551 ± 0.0263` | `0.4256 ± 0.0312` | `0.2236 ± 0.0462` | `0.8237 ± 0.0158` | `0.8576 ± 0.0315` | `0.3966 ± 0.0357` |

Compared with internal evaluation, this is a substantial drop:

- accuracy falls from `0.8454` to `0.5551`
- macro-F1 falls from `0.7265` to `0.4256`
- macro-AUROC falls from `0.9599` to `0.8237`

This is the clearest current evidence of distribution shift and should remain central to the final paper.

### 4.4 Calibration behavior

Internal temperature scaling produces only a small ECE gain in the exploratory internal slice:

- internal uncalibrated ECE: `0.0387 ± 0.0070`
- internal calibrated ECE: `0.0375 ± 0.0105`

On the external ISIC 2019 test set, temperature scaling fitted on the internal validation set does not transfer well:

- external uncalibrated ECE: `0.2011 ± 0.0372`
- external calibrated ECE: `0.2236 ± 0.0462`

This is an important trustworthy-ML result in its own right: calibration fitted in-distribution should not be assumed to remain reliable under external shift.

## 5. Discussion

The current repo already supports three defensible exploratory claims.

First, the internal HAM10000 pipeline is strong and stable enough to justify a serious applied paper path rather than a toy demo.

Second, the exploratory ablations are informative rather than cosmetic. They suggest that repair may help and that weighted sampling may not be beneficial in the present noisy-label setting, but they are not yet the final publication comparisons.

Third, the external validation result materially strengthens the manuscript. The drop on ISIC 2019 shows that the model is not simply “solved” and that trustworthiness claims should be made with explicit awareness of domain shift.

The corrected paper should therefore organize itself around three sharper claims:

- internal improvements do not guarantee external trustworthiness
- budgeted trusted-label repair must beat random repair, not just no repair
- calibration learned in-distribution may fail under external shift

## 6. Limitations

The current package should still be framed honestly:

- the current quoted numbers are exploratory results, not the final corrected reruns
- external validation is limited to one external dataset
- the external class mapping collapses `SCC` into `akiec` and filters `UNK`
- calibration transfer under shift is weak
- the manuscript does not yet include full paired significance testing
- there is no prospective clinical evaluation or reader study
- advanced extensions such as SelectiveNet, DivideMix, or SWAG are outside the verified slice

## 7. Conclusion

TrustQueryNet already supports a credible Q1-oriented upgrade path. The repo has strong exploratory internal results, a useful external validation slice, and now an implementation-ready corrected rerun family with explicit checkpoint-policy evaluation, deterministic repair scoring, dense selective metrics, a random-repair baseline, a clean-label upper bound, a robust-loss baseline scaffold, and overlap-audit tooling. The next step is not new method invention; it is disciplined rerunning and honest reporting.

## 8. Main-paper assets to include

Recommended main-paper assets after corrected reruns:

- main internal comparison table: trusted repair vs random repair vs no repair
- clean-label upper-bound / noisy-loss anchor table
- internal reliability diagram and risk-coverage curve for the corrected main model
- external reliability diagram and risk-coverage curve for the corrected main model
- internal-vs-external summary figure
- classwise external performance figure

Recommended supplementary assets:

- seed-level internal results table
- seed-level external validation table
- overlap-audit report
- dataset mapping note for ISIC 2019 labels
- exact reproduction commands and config paths
