# TrustQueryNet Portfolio Copy

## GitHub tagline

Trustworthy deep learning for dermatoscopic lesion classification under noisy supervision, budgeted trusted-label repair, calibration, and selective prediction.

## Short GitHub summary

TrustQueryNet is a research-oriented PyTorch pipeline for dermatoscopic lesion classification under noisy supervision. The final verified study combines lesion-level splits, class-dependent noise, budgeted trusted-label repair under simulated oracle supervision, calibration, selective prediction, multi-seed benchmarking, and external validation. In the corrected `5`-seed internal suite, the main repair configuration reached `0.8350 ± 0.0059` accuracy and `0.7152 ± 0.0216` macro-F1 on HAM10000, while external ISIC 2019 testing dropped to `0.5692 ± 0.0145` accuracy and `0.4427 ± 0.0117` macro-F1, reinforcing the project's central message that internal gains do not guarantee external trustworthiness.

## Resume bullets

- Built a trustworthy skin lesion classification pipeline in PyTorch for HAM10000, combining noisy-label training, budgeted trusted-label repair, calibration, and selective prediction in a reproducible Colab-ready workflow.
- Designed a modular experiment stack with persistent split manifests, noise manifests, metrics, plots, and export bundles, enabling reproducible multi-seed benchmarking, ablation studies, and external validation.
- Ran a corrected publication-style evidence suite with repair, no-repair, random-repair, clean-label, and robust-loss baselines, then quantified external generalization drop on the official ISIC 2019 test set.

## LinkedIn or portfolio paragraph

I built TrustQueryNet as a research-grade trustworthy ML project around dermatoscopic skin lesion classification. Instead of optimizing only top-line accuracy, the project explicitly handles noisy labels, limited trusted supervision, budgeted trusted-label repair under simulated oracle supervision, calibration, and selective prediction. In the corrected final study, the main `5`-seed repair configuration reached `0.8350 ± 0.0059` internal accuracy and `0.7152 ± 0.0216` macro-F1 on HAM10000, while external validation on ISIC 2019 fell to `0.5692 ± 0.0145` accuracy and `0.4427 ± 0.0117` macro-F1. That gap became part of the scientific contribution: internal improvements did not automatically translate to external trustworthiness.

## Statement of purpose paragraph

One project that best reflects how I like to work is TrustQueryNet, a trustworthy machine learning pipeline for dermatoscopic lesion classification under noisy supervision. I approached the problem as both an engineering and research challenge: I built lesion-level data handling for HAM10000, simulated class-dependent label corruption, added budgeted trusted-label repair and calibration modules, and exported the results into a reproducible paper package with multi-seed benchmarking, baseline comparisons, and external validation. In the final study, repair provided only modest gains over strong no-repair and random-repair baselines, and external testing on ISIC 2019 still showed a large generalization and calibration gap. That experience strengthened my interest in building ML systems that are not only accurate, but also reliable and honestly evaluated for high-stakes settings.

## Interview talking points

- I treated trustworthiness as a first-class objective, not an afterthought.
- I debugged a full-model class-collapse failure, then built a corrected evidence stack with explicit checkpoint-policy evaluation, random-repair baselines, robust-loss baselines, and external overlap auditing.
- I built the project so that local development, Colab training, artifact export, and paper packaging all used the same codebase.
- I can explain both the modeling choices and the engineering choices, including why lesion-level grouping, calibration, macro-F1, selective prediction, and external validation mattered for this dataset.
