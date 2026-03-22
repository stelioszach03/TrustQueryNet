# TrustQueryNet Paired Statistical Comparisons

Seed-paired comparisons were computed on the exported multiseed summaries. Each row reports the mean delta (left minus right), a paired bootstrap confidence interval with confidence 95%, and a two-sided exact sign-flip permutation p-value.

Because only five shared seeds are available for the main comparisons and three for the GCE anchor, the p-value resolution is intentionally coarse. The table should therefore be read primarily as an effect-size and uncertainty summary rather than as a thresholded significance screen.

## Internal Comparisons

| Comparison | Metric | Shared seeds | Delta (left - right) | 95% paired bootstrap CI | Exact sign-flip p |
| --- | --- | --- | ---: | --- | ---: |
| Repair vs no repair | Calibrated macro-F1 | `42,52,62,72,82` | 0.0007 | [-0.0231, 0.0217] | 1.0000 |
| Repair vs random repair | Calibrated macro-F1 | `42,52,62,72,82` | 0.0046 | [-0.0112, 0.0262] | 0.8750 |
| Repair vs GCE no repair | Calibrated macro-F1 | `42,52,62` | 0.5918 | [0.5552, 0.6332] | 0.2500 |
| Repair vs no repair | Calibrated accuracy | `42,52,62,72,82` | 0.0030 | [-0.0065, 0.0131] | 0.6250 |
| Repair vs random repair | Calibrated accuracy | `42,52,62,72,82` | 0.0031 | [-0.0034, 0.0105] | 0.6875 |
| Repair vs GCE no repair | Calibrated accuracy | `42,52,62` | 0.1565 | [0.1515, 0.1636] | 0.2500 |

## External Comparisons

| Comparison | Metric | Shared seeds | Delta (left - right) | 95% paired bootstrap CI | Exact sign-flip p |
| --- | --- | --- | ---: | --- | ---: |
| Repair vs no repair | Calibrated macro-F1 | `42,52,62,72,82` | 0.0139 | [-0.0044, 0.0281] | 0.1875 |
| Repair vs random repair | Calibrated macro-F1 | `42,52,62,72,82` | 0.0116 | [-0.0097, 0.0329] | 0.4375 |
| Repair vs no repair | Calibrated accuracy | `42,52,62,72,82` | 0.0062 | [-0.0059, 0.0183] | 0.3750 |
| Repair vs random repair | Calibrated accuracy | `42,52,62,72,82` | 0.0101 | [-0.0087, 0.0290] | 0.3750 |

## Notes

- Internal primary comparisons use calibrated test metrics from the locked `e12` publication recipe.
- Comparisons against `GCE no repair` are restricted to the shared three-seed subset (`42,52,62`), because that anchor was not run on five seeds.
- The main manuscript uses calibrated macro-F1 as the primary model-selection and reporting endpoint; calibrated accuracy is retained here as a secondary summary.
