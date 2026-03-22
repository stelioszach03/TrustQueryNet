# Internal Gains Do Not Guarantee External Trustworthiness in Dermatoscopic Classification Under Simulated Label Corruption

## Abstract

We present TrustQueryNet, an externally validated dermatoscopic classification study under simulated class-dependent label corruption, budgeted trusted-label repair under simulated oracle supervision, post-hoc calibration, and selective prediction. The pipeline combines lesion-level HAM10000 splits, persistent noise manifests, explicit best-checkpoint evaluation, multi-seed aggregation, and external testing on the official ISIC 2019 test set. With the corrected ConvNeXt-Tiny recipe, repair reached `0.8350 ± 0.0059` internal calibrated accuracy and `0.7152 ± 0.0216` internal macro-F1 across five seeds. However, strong baselines remained close: no repair reached `0.7145 ± 0.0130` internal macro-F1, random repair reached `0.7105 ± 0.0239`, and a clean-label upper bound reached `0.7330 ± 0.0288`. On external ISIC 2019, all methods degraded sharply. Repair reached `0.5692 ± 0.0145` calibrated accuracy and `0.4427 ± 0.0117` macro-F1, versus `0.5630 ± 0.0078` and `0.4288 ± 0.0123` for no repair, and `0.5591 ± 0.0203` and `0.4311 ± 0.0328` for random repair. Internal temperature scaling did not remove external calibration fragility, and generalized cross-entropy collapsed under the chosen corruption regime. An overlap audit found zero exact duplicate images between HAM10000 and the mapped ISIC 2019 external slice. Overall, the study shows that modest internal gains from trusted-label repair do not guarantee external trustworthiness and must be judged against strong baselines, including random repair.

## 1. Introduction

Dermatoscopic lesion classification is a useful test bed for trustworthy machine learning because it combines severe class imbalance, multiple images per lesion, dataset shift across collections, and imperfect labels. In this setting, a model can look strong in-distribution while remaining poorly calibrated or externally brittle. TrustQueryNet was developed to evaluate these issues directly rather than treating them as secondary diagnostics.

The project combines four ideas in one reproducible pipeline:

- lesion-level, group-aware splitting for HAM10000
- explicit class-dependent noisy-label simulation
- budgeted trusted-label repair under simulated oracle supervision
- calibration and selective prediction analysis under both internal and external evaluation

The final paper should not be framed as a new method paper. Its value is as a rigorous, externally validated trustworthy-ML study that asks a practical question: when labels are noisy and trusted correction is budget-limited, do modest internal gains from repair survive stronger baselines and external shift?

## 2. Methods

### 2.1 Data and split protocol

HAM10000 is the primary development dataset. TrustQueryNet performs lesion-level grouping so that images from the same lesion do not leak across train, validation, and test partitions. Split manifests are written to disk and reused across runs. Clean labels and observed labels are tracked separately, alongside trust and repair state.

The external dataset is the official ISIC 2019 test set. Labels are mapped into the HAM10000-style `7`-class space:

- `MEL -> mel`
- `NV -> nv`
- `BCC -> bcc`
- `AK -> akiec`
- `SCC -> akiec`
- `BKL -> bkl`
- `DF -> df`
- `VASC -> vasc`
- `UNK` filtered

### 2.2 Noise and repair protocol

Observed training labels are corrupted by a fixed class-dependent transition matrix. A fraction of samples is marked as initially trusted, and subsequent repair rounds replace `y_observed` with `y_clean` for selected samples. This is a retrospective simulation of limited expert correction, not a prospective clinician-in-the-loop study. The most accurate description is therefore **budgeted trusted-label repair under simulated oracle supervision**.

### 2.3 Models and evaluation

The final corrected study uses ConvNeXt-Tiny with:

- `12` epochs
- cross-entropy with `0.05` label smoothing for the main repair and comparison baselines
- explicit best-checkpoint selection by validation macro-F1
- temperature scaling
- dense selective metrics including AURC
- multi-seed aggregation

We report calibrated accuracy, macro-F1, macro-AUROC, ECE, AURC, coverage at confidence `0.5`, and risk at confidence `0.5`.

## 3. Experimental Protocol

The corrected internal evidence suite includes:

- `Repair`: budgeted entropy-based trusted-label repair
- `No repair`: identical training recipe without repair rounds
- `Random repair`: matched repair budget with random sample selection
- `Clean upper`: clean-label upper bound without simulated corruption
- `GCE no repair`: robust-loss baseline under the same noisy setup

The main external evidence suite evaluates the three most relevant operational variants:

- `Repair external`
- `No repair external`
- `Random repair external`

An image-overlap audit between HAM10000 and ISIC 2019 reports exact-hash matches and perceptual-hash near-duplicate candidates.

## 4. Results

### 4.1 Internal main comparison

| Setting | Accuracy | Macro-F1 | ECE | Macro-AUROC | AURC | Coverage@0.5 | Risk@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Repair` | `0.8350 ± 0.0059` | `0.7152 ± 0.0216` | `0.0445 ± 0.0090` | `0.9382 ± 0.0133` | `0.0728 ± 0.0186` | `0.9525 ± 0.0074` | `0.1428 ± 0.0054` |
| `No repair` | `0.8321 ± 0.0071` | `0.7145 ± 0.0130` | `0.0356 ± 0.0075` | `0.9460 ± 0.0128` | `0.0622 ± 0.0093` | `0.9480 ± 0.0066` | `0.1449 ± 0.0078` |
| `Random repair` | `0.8319 ± 0.0051` | `0.7105 ± 0.0239` | `0.0356 ± 0.0071` | `0.9521 ± 0.0128` | `0.0588 ± 0.0145` | `0.9376 ± 0.0100` | `0.1402 ± 0.0107` |

The internal comparison does not support a “repair wins everywhere” story. Repair produces a small point-performance gain in accuracy and macro-F1, but no-repair and random-repair remain extremely competitive, and both outperform repair on some trust metrics such as ECE, AUROC, and AURC.

Seed-paired reporting supports the same restrained reading. On calibrated macro-F1, repair is nearly indistinguishable from no repair (`delta = 0.0008`, `95% CI [-0.0231, 0.0217]`, exact sign-flip `p = 1.0000`) and only modestly above random repair (`delta = 0.0045`, `95% CI [-0.0112, 0.0262]`, `p = 0.8750`). Against GCE the gap is large on the shared three-seed subset, but the exact permutation p-value remains coarse because the anchor uses only three seeds.

### 4.2 Noisy-label anchors

| Setting | Accuracy | Macro-F1 | ECE | Macro-AUROC | AURC | Coverage@0.5 | Risk@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Repair` | `0.8350 ± 0.0059` | `0.7152 ± 0.0216` | `0.0445 ± 0.0090` | `0.9382 ± 0.0133` | `0.0728 ± 0.0186` | `0.9525 ± 0.0074` | `0.1428 ± 0.0054` |
| `Clean upper` | `0.8521 ± 0.0055` | `0.7330 ± 0.0288` | `0.0389 ± 0.0176` | `0.9583 ± 0.0076` | `0.0502 ± 0.0095` | `0.9618 ± 0.0107` | `0.1306 ± 0.0089` |
| `GCE no repair` | `0.6774 ± 0.0012` | `0.1224 ± 0.0122` | `0.0268 ± 0.0381` | `0.5782 ± 0.1279` | `0.2399 ± 0.1059` | `0.9639 ± 0.0626` | `0.3044 ± 0.0327` |

The clean upper bound shows moderate remaining headroom, while the generalized cross-entropy baseline fails dramatically under the chosen noisy-label regime. This negative result is important: robust-loss substitution is not automatically beneficial in this setting.

### 4.3 External validation

| Setting | Accuracy | Macro-F1 | ECE | Macro-AUROC | AURC | Coverage@0.5 | Risk@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Repair external` | `0.5692 ± 0.0145` | `0.4427 ± 0.0117` | `0.2000 ± 0.0253` | `0.8125 ± 0.0180` | `0.2804 ± 0.0299` | `0.8526 ± 0.0230` | `0.3805 ± 0.0228` |
| `No repair external` | `0.5630 ± 0.0078` | `0.4288 ± 0.0123` | `0.2016 ± 0.0123` | `0.8168 ± 0.0154` | `0.2749 ± 0.0183` | `0.8498 ± 0.0264` | `0.3838 ± 0.0073` |
| `Random repair external` | `0.5591 ± 0.0203` | `0.4311 ± 0.0328` | `0.2149 ± 0.0395` | `0.8114 ± 0.0337` | `0.2905 ± 0.0669` | `0.8654 ± 0.0383` | `0.3964 ± 0.0269` |

All methods degrade sharply on ISIC 2019 relative to internal HAM10000 evaluation. Repair remains modestly ahead of the two comparison baselines on external accuracy and macro-F1, but does not dominate all trust metrics. The overall lesson is not that repair solves external robustness, but that even modest internal gains must be interpreted cautiously under shift.

The external paired comparisons remain modest rather than decisive. On calibrated macro-F1, repair exceeds no repair by `0.0139` (`95% CI [-0.0044, 0.0281]`, exact sign-flip `p = 0.1875`) and random repair by `0.0117` (`95% CI [-0.0097, 0.0329]`, `p = 0.4375`).

### 4.4 Overlap audit

The overlap audit found:

- `10015` HAM10000 samples
- `6191` external ISIC 2019 samples after label filtering/mapping
- `0` exact duplicate images
- `14185` perceptual-hash candidate pairs at `dHash <= 4`

The zero exact matches are the key integrity result. The perceptual-hash list is a coarse candidate screen for manual or secondary review and should not be interpreted as confirmed leakage.

## 5. Discussion

The final evidence supports a restrained but useful conclusion. Budgeted trusted-label repair is not a universally dominant intervention in this setting. It yields only a modest advantage over no-repair and random-repair baselines in point performance, while other trust metrics remain mixed. This is exactly why the stronger baseline suite matters: without random repair and no repair, it would have been easy to overstate the value of repair.

The external findings are the most important part of the paper. Internal HAM10000 performance is strong for all competitive variants, but all of them degrade substantially on ISIC 2019. That drop is not a side note; it is the central trustworthy-ML result. Similarly, post-hoc calibration learned in-distribution does not remove external confidence fragility.

The paper is therefore strongest when framed as an externally validated experimental study rather than a method paper. Its contribution is evidence: which interventions remain competitive, which fail, and how internal trust signals do or do not transfer externally.

## 6. Limitations

- The label-corruption process is simulated and class-dependent rather than estimated from observed expert disagreement.
- The repair protocol is simulated oracle supervision, not prospective expert annotation.
- External validation is limited to one external dataset.
- Paired significance reporting is available only for the prespecified main comparisons and is resolution-limited by the number of seeds.
- Perceptual-hash overlap candidates were not exhaustively adjudicated manually, so only the zero exact-duplicate result is definitive.
- No clinician reader study or deployment evaluation is included.
- The work does not introduce a new learning algorithm.

## 7. Conclusion

TrustQueryNet is now a credible applied research package for trustworthy dermatoscopic classification under noisy supervision. The final evidence shows that internal gains do not guarantee external trustworthiness, that trusted-label repair must be judged against strong baselines including random repair, and that robust-loss substitution can fail outright in this regime. These are publishable applied findings when presented honestly and with external validation.
