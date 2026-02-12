# TrustQueryNet Portfolio Copy

## GitHub tagline

Trustworthy deep learning for dermatoscopic lesion classification under noisy supervision, active queries, calibration, and selective prediction.

## Short GitHub summary

TrustQueryNet is a research-oriented PyTorch pipeline for dermatoscopic lesion classification under noisy supervision. The final paper-facing configuration combines group-aware lesion splits, class-dependent noise, active querying, calibration, and selective risk analysis, reaching `0.8454 ± 0.0054` internal accuracy, `0.7265 ± 0.0080` macro-F1, and `0.9599 ± 0.0048` macro-AUROC across `5` HAM10000 seeds, plus external validation on the official ISIC 2019 test set.

## Resume bullets

- Built a trustworthy skin lesion classification pipeline in PyTorch for HAM10000, combining noisy-label training, active querying, calibration, and selective prediction in a reproducible Colab-ready workflow.
- Designed a modular experiment stack with persistent split manifests, noise manifests, metrics, plots, and export bundles, enabling reproducible multi-seed benchmarking, ablation studies, and external validation.
- Improved the HAM10000 study from a pilot baseline of `0.4950` macro-F1 to a multi-seed `0.7265 ± 0.0080` macro-F1 internal result, then quantified generalization drop on the official ISIC 2019 external test set.

## LinkedIn or portfolio paragraph

I built TrustQueryNet as a research-grade trustworthy ML project around dermatoscopic skin lesion classification. Instead of optimizing only top-line accuracy, the project explicitly handles noisy labels, limited trusted supervision, uncertainty-aware active querying, calibration, and selective prediction. The final paper-facing model reached `0.8454 ± 0.0054` internal accuracy, `0.7265 ± 0.0080` macro-F1, and `0.9599 ± 0.0048` macro-AUROC across `5` HAM10000 seeds, and I extended the study with an external ISIC 2019 validation slice to quantify domain shift.

## Statement of purpose paragraph

One project that best reflects how I like to work is TrustQueryNet, a trustworthy machine learning pipeline for dermatoscopic lesion classification under noisy supervision. I approached the problem as both an engineering and research challenge: I built group-aware data handling for HAM10000, simulated clinically plausible label noise, added active querying and calibration modules, and exported the results into a reproducible paper package with multi-seed benchmarking, ablation studies, and external validation. The final ConvNeXt-Tiny study reached `0.7265 ± 0.0080` macro-F1 and `0.9599 ± 0.0048` macro-AUROC internally, while external testing on ISIC 2019 exposed a meaningful generalization drop that became part of the core scientific story. That experience strengthened my interest in building ML systems that are not only accurate, but also reliable and honestly evaluated for high-stakes settings.

## Interview talking points

- I treated trustworthiness as a first-class objective, not an afterthought.
- I debugged a full-model class-collapse failure, then used multi-seed ablations to show that removing weighted sampling actually produced the strongest final model.
- I built the project so that local development, Colab training, artifact export, and paper packaging all used the same codebase.
- I can explain both the modeling choices and the engineering choices, including why lesion-level grouping, calibration, macro-F1, and external validation mattered for this dataset.
