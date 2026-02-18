"""Helpers for Q1 rerun phase-gate verification."""

from __future__ import annotations

from typing import Any


def _same_training_family(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = [
        "backbone",
        "epochs",
        "batch_size",
        "lr",
        "loss",
        "sampler",
        "amp",
        "warmup_epochs",
        "early_stopping_patience",
    ]
    return all(left["training"].get(key) == right["training"].get(key) for key in keys)


def _same_noise(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left.get("noise") == right.get("noise")


def _same_evaluation(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left.get("evaluation") == right.get("evaluation")


def verify_config_gate(repair_cfg: dict[str, Any], no_repair_cfg: dict[str, Any], random_cfg: dict[str, Any]) -> dict[str, bool]:
    repair_active = repair_cfg.get("active_learning", {})
    no_repair_active = no_repair_cfg.get("active_learning", {})
    random_active = random_cfg.get("active_learning", {})

    return {
        "no_repair_deconfounded": (
            _same_training_family(repair_cfg, no_repair_cfg)
            and _same_noise(repair_cfg, no_repair_cfg)
            and _same_evaluation(repair_cfg, no_repair_cfg)
            and repair_active.get("initial_clean_fraction") == no_repair_active.get("initial_clean_fraction")
            and no_repair_active.get("query_size") == 0
        ),
        "random_repair_matches_budget": (
            _same_training_family(repair_cfg, random_cfg)
            and _same_noise(repair_cfg, random_cfg)
            and _same_evaluation(repair_cfg, random_cfg)
            and repair_active.get("query_size") == random_active.get("query_size")
            and repair_active.get("rounds") == random_active.get("rounds")
            and repair_active.get("initial_clean_fraction") == random_active.get("initial_clean_fraction")
            and random_active.get("method") == "random"
        ),
    }
