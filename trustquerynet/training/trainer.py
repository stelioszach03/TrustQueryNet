"""Training orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import yaml
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader

from trustquerynet.data.cifar100 import prepare_cifar100_splits
from trustquerynet.data.ham10000_isic import prepare_ham10000_splits
from trustquerynet.eval.calibration import expected_calibration_error
from trustquerynet.eval.metrics import accuracy_from_probs, compute_all, macro_f1_from_probs
from trustquerynet.eval.plots import save_reliability_diagram, save_risk_coverage_plot
from trustquerynet.eval.selective import default_threshold_grid, risk_coverage_curve
from trustquerynet.eval.stats_tests import bootstrap_metric_ci
from trustquerynet.methods.losses import build_loss
from trustquerynet.models.backbones import create_backbone, forward_with_embeddings
from trustquerynet.noise.base import build_noise_model
from trustquerynet.training.checkpointing import save_checkpoint
from trustquerynet.training.reproducibility import choose_device, set_seed
from trustquerynet.uncertainty.mc_dropout import predict_mc_dropout
from trustquerynet.uncertainty.temperature_scaling import fit_temperature


@dataclass
class RunArtifacts:
    output_dir: str
    metrics: Dict[str, Any]
    checkpoint_paths: Dict[str, str]
    train_probs: np.ndarray
    train_logits: np.ndarray
    train_labels: np.ndarray
    train_embeddings: np.ndarray | None
    train_mc_probs: np.ndarray | None
    test_probs: np.ndarray


def build_dataset_bundle(cfg):
    dataset_cfg = cfg["dataset"]
    name = dataset_cfg["name"].lower()
    if name == "cifar100":
        return prepare_cifar100_splits(
            root=dataset_cfg["root"],
            seed=int(cfg["seed"]),
            val_ratio=float(dataset_cfg.get("val_ratio", 0.15)),
            img_size=int(dataset_cfg["img_size"]),
            max_train_samples=dataset_cfg.get("max_train_samples"),
            max_val_samples=dataset_cfg.get("max_val_samples"),
            max_test_samples=dataset_cfg.get("max_test_samples"),
        )
    if name == "ham10000":
        return prepare_ham10000_splits(
            metadata_csv=dataset_cfg["metadata_csv"],
            image_dir=dataset_cfg["image_dir"],
            seed=int(cfg["seed"]),
            ratios={
                "train": 1.0 - float(dataset_cfg.get("val_ratio", 0.15)) - float(dataset_cfg.get("test_ratio", 0.15)),
                "val": float(dataset_cfg.get("val_ratio", 0.15)),
                "test": float(dataset_cfg.get("test_ratio", 0.15)),
            },
            img_size=int(dataset_cfg["img_size"]),
            max_train_samples=dataset_cfg.get("max_train_samples"),
            max_val_samples=dataset_cfg.get("max_val_samples"),
            max_test_samples=dataset_cfg.get("max_test_samples"),
            split_csv=dataset_cfg.get("split_csv"),
            save_split_csv=dataset_cfg.get("save_split_csv"),
        )
    raise ValueError(f"Unsupported dataset: {name}")


def initialize_train_noise(cfg, train_dataset, num_classes: int):
    clean_labels = train_dataset.get_clean_labels()
    noise_cfg = cfg.get("noise", {})
    noise_model = build_noise_model(noise_cfg, num_classes=num_classes)
    observed_labels, noise_info = noise_model.generate(clean_labels, seed=int(cfg["seed"]))
    train_dataset.set_observed_labels(observed_labels)
    return observed_labels, noise_info


def _create_optimizer(cfg, model):
    training_cfg = cfg["training"]
    lr = float(training_cfg["lr"])
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    name = training_cfg.get("optimizer", "adamw").lower()
    if name == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    raise ValueError(f"Unsupported optimizer: {name}")


def _build_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, device: torch.device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def _run_epoch(model, loader, criterion, optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["y_observed"].to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
    return total_loss / max(total_examples, 1)


@torch.no_grad()
def _collect_predictions(
    model,
    loader,
    criterion,
    device: torch.device,
    target_key: str = "y_clean",
    return_embeddings: bool = False,
) -> Dict[str, Any]:
    model.eval()
    losses = []
    logits_list = []
    probs_list = []
    labels_list = []
    indices_list = []
    embeddings_list = []
    for batch in loader:
        images = batch["image"].to(device)
        targets = batch[target_key].to(device)
        logits, embeddings = forward_with_embeddings(model, images)
        loss = criterion(logits, targets)
        probs = torch.softmax(logits, dim=1)
        losses.append(loss.item() * images.size(0))
        logits_list.append(logits.cpu())
        probs_list.append(probs.cpu())
        labels_list.append(batch[target_key].cpu())
        indices_list.append(batch["index"].cpu())
        if return_embeddings:
            embeddings_list.append(embeddings.cpu())
    logits = torch.cat(logits_list).numpy()
    probs = torch.cat(probs_list).numpy()
    labels = torch.cat(labels_list).numpy()
    indices = torch.cat(indices_list).numpy()
    embeddings = torch.cat(embeddings_list).numpy() if embeddings_list else None
    total_examples = len(labels)
    return {
        "loss": float(sum(losses) / max(total_examples, 1)),
        "logits": logits,
        "probs": probs,
        "labels": labels,
        "indices": indices,
        "embeddings": embeddings,
    }


def _write_manifests(output_dir: Path, bundle, noise_info: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    split_frames = []
    for split_name, manifest in bundle.manifests.items():
        frame = manifest.copy()
        frame["split"] = split_name
        split_frames.append(frame)
    pd.concat(split_frames, ignore_index=True).to_csv(output_dir / "splits.csv", index=False)
    with (output_dir / "noise_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(noise_info, handle, indent=2)
    with (output_dir / "run_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)


def _compute_bootstrap_summary(y_true: np.ndarray, probs: np.ndarray, eval_cfg: Dict[str, Any]) -> Dict[str, Any]:
    n_bootstrap = int(eval_cfg.get("bootstrap_samples", 0))
    if n_bootstrap <= 0:
        return {}
    bootstrap_seed = int(eval_cfg.get("bootstrap_seed", 42))
    return {
        "accuracy": bootstrap_metric_ci(
            y_true,
            probs,
            metric_fn=accuracy_from_probs,
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed,
        ),
        "macro_f1": bootstrap_metric_ci(
            y_true,
            probs,
            metric_fn=macro_f1_from_probs,
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed + 1,
        ),
        "ece": bootstrap_metric_ci(
            y_true,
            probs,
            metric_fn=lambda y, p: expected_calibration_error(y, p),
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed + 2,
        ),
    }


def train_one_run(cfg, dataset_bundle=None, output_dir=None, apply_noise: bool = True) -> RunArtifacts:
    set_seed(int(cfg["seed"]), deterministic=bool(cfg.get("deterministic", False)))
    device = choose_device(cfg.get("device", "auto"))
    output_dir = Path(output_dir or cfg["output_dir"])

    bundle = dataset_bundle or build_dataset_bundle(cfg)
    num_classes = len(bundle.class_names)
    noise_info = {"noise_type": "pre_applied", "seed": int(cfg["seed"])}
    if apply_noise:
        _, noise_info = initialize_train_noise(cfg, bundle.train, num_classes)
        bundle.manifests["train"]["y_observed"] = bundle.train.manifest["y_observed"].to_numpy()
    else:
        noise_info["realized_flip_rate"] = float(
            np.mean(bundle.train.get_observed_labels() != bundle.train.get_clean_labels())
        )
        bundle.manifests["train"]["y_observed"] = bundle.train.manifest["y_observed"].to_numpy()

    _write_manifests(output_dir, bundle, noise_info, cfg)

    model = create_backbone(
        name=cfg["training"]["backbone"],
        pretrained=bool(cfg["training"].get("pretrained", True)),
        num_classes=num_classes,
        img_size=int(cfg["dataset"]["img_size"]),
    ).to(device)

    criterion = build_loss(
        cfg["training"]["loss"],
        label_smoothing=float(cfg["training"].get("label_smoothing", 0.0)),
        num_classes=num_classes,
    )
    optimizer = _create_optimizer(cfg, model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(cfg["training"]["epochs"]), 1))

    train_loader = _build_loader(
        bundle.train,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 2)),
        device=device,
    )
    val_loader = _build_loader(
        bundle.val,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 2)),
        device=device,
    )
    test_loader = _build_loader(
        bundle.test,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 2)),
        device=device,
    )

    best_val_loss = float("inf")
    best_macro_f1 = float("-inf")
    best_ece = float("inf")
    checkpoint_paths = {
        "last": str(output_dir / "last.ckpt"),
        "best_val_loss": str(output_dir / "best_val_loss.ckpt"),
        "best_val_macro_f1": str(output_dir / "best_val_macro_f1.ckpt"),
        "best_val_ece": str(output_dir / "best_val_ece.ckpt"),
    }

    history = []
    for epoch in range(int(cfg["training"]["epochs"])):
        train_loss = _run_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        val_outputs = _collect_predictions(model, val_loader, criterion, device, target_key="y_clean")
        val_metrics = compute_all(
            y_true=val_outputs["labels"],
            probs=val_outputs["probs"],
            thresholds=cfg.get("evaluation", {}).get("thresholds", [0.5]),
        )
        val_metrics["loss"] = val_outputs["loss"]
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val": val_metrics})

        save_checkpoint(checkpoint_paths["last"], model, optimizer, epoch + 1, extra={"history": history})
        if val_outputs["loss"] < best_val_loss:
            best_val_loss = val_outputs["loss"]
            save_checkpoint(checkpoint_paths["best_val_loss"], model, optimizer, epoch + 1, extra={"metric": "loss"})
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            save_checkpoint(
                checkpoint_paths["best_val_macro_f1"],
                model,
                optimizer,
                epoch + 1,
                extra={"metric": "macro_f1"},
            )
        if val_metrics["ece"] < best_ece:
            best_ece = val_metrics["ece"]
            save_checkpoint(checkpoint_paths["best_val_ece"], model, optimizer, epoch + 1, extra={"metric": "ece"})

    train_eval_loader = _build_loader(
        bundle.train,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 2)),
        device=device,
    )
    train_outputs = _collect_predictions(
        model,
        train_eval_loader,
        criterion,
        device,
        target_key="y_clean",
        return_embeddings=True,
    )
    val_outputs = _collect_predictions(model, val_loader, criterion, device, target_key="y_clean")
    test_outputs = _collect_predictions(model, test_loader, criterion, device, target_key="y_clean")

    uncertainty_cfg = cfg.get("uncertainty", {})
    uncertainty_method = uncertainty_cfg.get("method", "softmax").lower()
    mc_samples = int(uncertainty_cfg.get("mc_dropout_samples", 8))
    need_train_mc = uncertainty_method == "mc_dropout" or (
        cfg.get("active_learning", {}).get("enabled", False)
        and cfg.get("active_learning", {}).get("method", "").lower() == "bald"
    )
    train_mc_outputs = None
    test_mc_outputs = None
    if need_train_mc:
        train_mc_outputs = predict_mc_dropout(model, train_eval_loader, device, num_samples=mc_samples)
    if uncertainty_method == "mc_dropout":
        test_mc_outputs = predict_mc_dropout(model, test_loader, device, num_samples=mc_samples)

    scaler = fit_temperature(val_outputs["logits"], val_outputs["labels"])
    calibrated_test_probs = (
        scaler.predict_proba(torch.tensor(test_outputs["logits"], dtype=torch.float32)).detach().numpy()
    )
    test_probs_for_metrics = test_mc_outputs["mean_probs"] if test_mc_outputs is not None else test_outputs["probs"]

    thresholds = cfg.get("evaluation", {}).get("thresholds")
    if thresholds is None:
        thresholds = default_threshold_grid().tolist()

    test_metrics = compute_all(test_outputs["labels"], test_probs_for_metrics, thresholds=thresholds)
    calibrated_metrics = compute_all(test_outputs["labels"], calibrated_test_probs, thresholds=thresholds)
    rc_uncal = risk_coverage_curve(test_outputs["labels"], test_probs_for_metrics, thresholds)
    rc_cal = risk_coverage_curve(test_outputs["labels"], calibrated_test_probs, thresholds)

    save_reliability_diagram(output_dir / "plots" / "reliability_uncalibrated.png", test_outputs["labels"], test_probs_for_metrics, "Reliability Diagram (Uncalibrated)")
    save_reliability_diagram(output_dir / "plots" / "reliability_calibrated.png", test_outputs["labels"], calibrated_test_probs, "Reliability Diagram (Temperature Scaled)")
    save_risk_coverage_plot(output_dir / "plots" / "risk_coverage_uncalibrated.png", rc_uncal, "Risk-Coverage (Uncalibrated)")
    save_risk_coverage_plot(output_dir / "plots" / "risk_coverage_calibrated.png", rc_cal, "Risk-Coverage (Temperature Scaled)")

    eval_cfg = cfg.get("evaluation", {})
    metrics = {
        "device": str(device),
        "history": history,
        "noise": noise_info,
        "uncertainty_method": uncertainty_method,
        "test_uncalibrated": test_metrics,
        "test_calibrated": calibrated_metrics,
    }
    bootstrap_summary = _compute_bootstrap_summary(test_outputs["labels"], calibrated_test_probs, eval_cfg)
    if bootstrap_summary:
        metrics["bootstrap_ci_calibrated"] = bootstrap_summary
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return RunArtifacts(
        output_dir=str(output_dir),
        metrics=metrics,
        checkpoint_paths=checkpoint_paths,
        train_probs=train_outputs["probs"],
        train_logits=train_outputs["logits"],
        train_labels=train_outputs["labels"],
        train_embeddings=train_outputs["embeddings"],
        train_mc_probs=train_mc_outputs["samples"] if train_mc_outputs is not None else None,
        test_probs=test_probs_for_metrics,
    )
