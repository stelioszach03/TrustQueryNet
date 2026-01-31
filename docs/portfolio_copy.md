# TrustQueryNet Portfolio Copy

## GitHub tagline

Trustworthy deep learning for dermatoscopic lesion classification under noisy supervision, active queries, calibration, and selective prediction.

## Short GitHub summary

TrustQueryNet is a research-oriented PyTorch pipeline for HAM10000 that studies what happens when labels are imperfect and model confidence matters. The final verified configuration combines group-aware lesion splits, class-dependent noise, active querying, calibration, and selective risk analysis, reaching `0.8330` calibrated test accuracy, `0.7372` macro-F1, and `0.9434` macro-AUROC on HAM10000.

## Resume bullets

- Built a trustworthy skin lesion classification pipeline in PyTorch for HAM10000, combining noisy-label training, active querying, calibration, and selective prediction in a reproducible Colab-ready workflow.
- Designed a modular experiment stack with persistent split manifests, noise manifests, metrics, plots, and export bundles, enabling reproducible pilot-to-full comparisons for research-style reporting.
- Improved the final HAM10000 result from a pilot baseline of `0.4950` macro-F1 to `0.7372` macro-F1 and reduced calibrated ECE to `0.0250` using a balanced ConvNeXt-Tiny configuration.

## LinkedIn or portfolio paragraph

I built TrustQueryNet as a research-grade trustworthy ML project around dermatoscopic skin lesion classification. Instead of optimizing only top-line accuracy, the project explicitly handles noisy labels, limited trusted supervision, uncertainty-aware active querying, calibration, and selective prediction. The final exported HAM10000 run achieved `0.8330` calibrated test accuracy, `0.7372` macro-F1, and `0.9434` macro-AUROC, with a clean artifact pipeline for figures, tables, and reproducible write-ups.

## Statement of purpose paragraph

One project that best reflects how I like to work is TrustQueryNet, a trustworthy machine learning pipeline for dermatoscopic lesion classification under noisy supervision. I approached the problem as both an engineering and research challenge: I built group-aware data handling for HAM10000, simulated clinically plausible label noise, added active querying and calibration modules, and exported the results into a reproducible paper bundle. The final balanced ConvNeXt-Tiny configuration improved the project from a pilot `0.4950` macro-F1 baseline to `0.7372` macro-F1 with `0.9434` macro-AUROC and low calibration error. That experience strengthened my interest in building ML systems that are not only accurate, but also reliable and interpretable enough for high-stakes settings.

## Interview talking points

- I treated trustworthiness as a first-class objective, not an afterthought.
- I debugged a full-model class-collapse failure and stabilized the final run with weighted sampling and a safer training recipe.
- I built the project so that local development, Colab training, artifact export, and paper packaging all used the same codebase.
- I can explain both the modeling choices and the engineering choices, including why lesion-level grouping, calibration, and macro-F1 mattered for this dataset.
