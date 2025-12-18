"""Simple active learning orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from trustquerynet.active.acquisition import select_query_indices


@dataclass
class ActiveLearningReport:
    output_dir: str
    initial_trusted_indices: list[int]
    selected_indices_by_round: list[list[int]]
    round_metrics: list[dict]
    final_metrics: dict


def _select_initial_trusted_indices(labels: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    if fraction <= 0.0:
        return np.asarray([], dtype=np.int64)
    indices = np.arange(len(labels))
    target_count = int(round(len(indices) * fraction))
    if target_count <= 0:
        return np.asarray([], dtype=np.int64)
    target_count = min(target_count, len(indices))

    label_counts = np.bincount(labels)
    positive_counts = label_counts[label_counts > 0]
    can_stratify = len(positive_counts) > 1 and positive_counts.min() >= 2 and target_count >= len(positive_counts)
    trusted_indices, _ = train_test_split(
        indices,
        train_size=target_count,
        random_state=seed,
        stratify=labels if can_stratify else None,
    )
    return np.sort(trusted_indices.astype(np.int64))


def run_active_learning(cfg) -> ActiveLearningReport:
    from trustquerynet.training.trainer import build_dataset_bundle, initialize_train_noise, train_one_run

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = build_dataset_bundle(cfg)
    class_count = len(bundle.class_names)
    initialize_train_noise(cfg, bundle.train, class_count)
    initial_clean_fraction = float(cfg["active_learning"].get("initial_clean_fraction", 0.0))
    initial_trusted_indices = _select_initial_trusted_indices(
        bundle.train.get_clean_labels(),
        fraction=initial_clean_fraction,
        seed=int(cfg["seed"]),
    )
    bundle.train.mark_trusted(initial_trusted_indices)
    bundle.manifests["train"]["y_observed"] = bundle.train.manifest["y_observed"].to_numpy()
    bundle.manifests["train"]["is_trusted"] = bundle.train.manifest["is_trusted"].to_numpy()

    rounds = int(cfg["active_learning"].get("rounds", 1))
    query_size = int(cfg["active_learning"].get("query_size", 0))
    method = cfg["active_learning"].get("method", "entropy")

    selected_indices_by_round = []
    round_metrics = []
    selected_mask = (
        bundle.train.manifest["is_queried"].to_numpy(dtype=bool, copy=True)
        | bundle.train.manifest["is_trusted"].to_numpy(dtype=bool, copy=True)
    )
    final_metrics = {}
    shortlist_factor = int(cfg["active_learning"].get("shortlist_factor", 4))

    for round_idx in range(rounds):
        round_dir = output_dir / f"active_round_{round_idx + 1}"
        artifacts = train_one_run(cfg, dataset_bundle=bundle, output_dir=round_dir, apply_noise=False)
        round_metrics.append(artifacts.metrics)
        final_metrics = artifacts.metrics

        if round_idx == rounds - 1 or query_size <= 0:
            continue

        candidate_indices = np.where(~selected_mask)[0]
        if len(candidate_indices) == 0:
            continue
        top_global = select_query_indices(
            method,
            budget=query_size,
            probs=artifacts.train_probs,
            samples=artifacts.train_mc_probs,
            embeddings=artifacts.train_embeddings,
            selected_mask=selected_mask,
            shortlist_factor=shortlist_factor,
        )
        bundle.train.repair_labels(top_global)
        selected_mask[top_global] = True
        selected_indices_by_round.append(top_global.tolist())

    report = ActiveLearningReport(
        output_dir=str(output_dir),
        initial_trusted_indices=initial_trusted_indices.tolist(),
        selected_indices_by_round=selected_indices_by_round,
        round_metrics=round_metrics,
        final_metrics=final_metrics,
    )
    report_path = output_dir / "active_learning_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "initial_trusted_indices": initial_trusted_indices.tolist(),
                "selected_indices_by_round": selected_indices_by_round,
                "round_metrics": round_metrics,
                "final_metrics": final_metrics,
            },
            handle,
            indent=2,
        )
    return report
