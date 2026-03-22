# TrustQueryNet Config Families

This directory intentionally keeps multiple config families because they serve different purposes in the repo.

## Final paper-facing configs

These are the configs that define the corrected TrustQueryNet evidence slice described in the manuscript:

- `q1_ham10000_convnext_repair.yaml`
- `q1_ham10000_convnext_no_repair.yaml`
- `q1_ham10000_convnext_random_repair.yaml`
- `q1_ham10000_convnext_clean_upper.yaml`
- `q1_ham10000_convnext_gce_no_repair.yaml`

These are the public paper-critical configs. The locked publication recipe is the `e12` ConvNeXt-Tiny slice derived from this family.

## Supporting paper-adjacent config

- `q1_ham10000_convnext_weighted_secondary.yaml`

This is a retained secondary comparison config. It is not part of the main paper evidence slice.

## Verified development configs kept for provenance

- `quick_cifar100.yaml`
- `quick_cifar100_active.yaml`
- `pilot_ham10000.yaml`
- `full_ham10000_convnext.yaml`
- `full_ham10000_convnext_no_querying.yaml`
- `full_ham10000_convnext_no_weighted.yaml`

These older configs remain in the repo because they document verified development stages and smoke or pilot workflows. They should not be confused with the final paper-facing evidence slice.

## Naming guide

- `quick_*`: lightweight smoke checks
- `pilot_*`: small early HAM10000 validation slice
- `full_*`: earlier full HAM10000 development-era runs preserved for provenance
- `q1_*`: corrected paper-oriented comparison family
