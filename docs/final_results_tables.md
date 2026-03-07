# TrustQueryNet Final Results Tables

## Internal Main Comparison

| Setting | Accuracy | Macro-F1 | ECE | Macro-AUROC | AURC | Coverage@0.5 | Risk@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Repair` | `0.8350 ± 0.0059` | `0.7152 ± 0.0216` | `0.0445 ± 0.0090` | `0.9382 ± 0.0133` | `0.0728 ± 0.0186` | `0.9525 ± 0.0074` | `0.1428 ± 0.0054` |
| `No repair` | `0.8321 ± 0.0071` | `0.7145 ± 0.0130` | `0.0356 ± 0.0075` | `0.9460 ± 0.0128` | `0.0622 ± 0.0093` | `0.9480 ± 0.0066` | `0.1449 ± 0.0078` |
| `Random repair` | `0.8319 ± 0.0051` | `0.7105 ± 0.0239` | `0.0356 ± 0.0071` | `0.9521 ± 0.0128` | `0.0588 ± 0.0145` | `0.9376 ± 0.0100` | `0.1402 ± 0.0107` |

## Noisy-Label Anchors

| Setting | Accuracy | Macro-F1 | ECE | Macro-AUROC | AURC | Coverage@0.5 | Risk@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Repair` | `0.8350 ± 0.0059` | `0.7152 ± 0.0216` | `0.0445 ± 0.0090` | `0.9382 ± 0.0133` | `0.0728 ± 0.0186` | `0.9525 ± 0.0074` | `0.1428 ± 0.0054` |
| `Clean upper` | `0.8521 ± 0.0055` | `0.7330 ± 0.0288` | `0.0389 ± 0.0176` | `0.9583 ± 0.0076` | `0.0502 ± 0.0095` | `0.9618 ± 0.0107` | `0.1306 ± 0.0089` |
| `GCE no repair` | `0.6774 ± 0.0012` | `0.1224 ± 0.0122` | `0.0268 ± 0.0381` | `0.5782 ± 0.1279` | `0.2399 ± 0.1059` | `0.9639 ± 0.0626` | `0.3044 ± 0.0327` |

## External Main Comparison

| Setting | Accuracy | Macro-F1 | ECE | Macro-AUROC | AURC | Coverage@0.5 | Risk@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Repair external` | `0.5692 ± 0.0145` | `0.4427 ± 0.0117` | `0.2000 ± 0.0253` | `0.8125 ± 0.0180` | `0.2804 ± 0.0299` | `0.8526 ± 0.0230` | `0.3805 ± 0.0228` |
| `No repair external` | `0.5630 ± 0.0078` | `0.4288 ± 0.0123` | `0.2016 ± 0.0123` | `0.8168 ± 0.0154` | `0.2749 ± 0.0183` | `0.8498 ± 0.0264` | `0.3838 ± 0.0073` |
| `Random repair external` | `0.5591 ± 0.0203` | `0.4311 ± 0.0328` | `0.2149 ± 0.0395` | `0.8114 ± 0.0337` | `0.2905 ± 0.0669` | `0.8654 ± 0.0383` | `0.3964 ± 0.0269` |

## Overlap Audit Summary

| Audit item | Value |
| --- | ---: |
| HAM10000 samples | `10015` |
| ISIC 2019 mapped external samples | `6191` |
| Exact duplicate images | `0` |
| dHash near-duplicate candidate pairs (`<= 4`) | `14185` |

Interpretation:

- The `0` exact duplicate count is the key data-integrity result.
- The dHash candidate list is a coarse screen and should not be treated as confirmed overlap without manual adjudication.
