"""Reproducible split helpers."""

from __future__ import annotations

from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split


def make_group_stratified_split(
    df: pd.DataFrame,
    label_col: str,
    group_col: str,
    seed: int,
    ratios: Dict[str, float],
) -> pd.DataFrame:
    """Split a dataframe by group while preserving label proportions approximately."""

    train_ratio = ratios.get("train", 0.7)
    val_ratio = ratios.get("val", 0.15)
    test_ratio = ratios.get("test", 0.15)
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0.")

    grouped = (
        df[[group_col, label_col]]
        .drop_duplicates()
        .groupby(group_col, as_index=False)
        .agg({label_col: "first"})
    )
    consistency = df.groupby(group_col)[label_col].nunique()
    if (consistency > 1).any():
        bad_groups = consistency[consistency > 1].index.tolist()[:5]
        raise ValueError(f"Each group must map to a single label. Bad groups: {bad_groups}")

    train_groups, temp_groups = train_test_split(
        grouped[group_col].to_numpy(),
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=grouped[label_col].to_numpy(),
    )

    temp_df = grouped[grouped[group_col].isin(temp_groups)].copy()
    relative_test = test_ratio / (val_ratio + test_ratio)
    temp_labels = temp_df[label_col].to_numpy()
    temp_class_counts = pd.Series(temp_labels).value_counts()
    can_stratify_temp = len(temp_df) >= 2 and (temp_class_counts.min() >= 2)
    val_groups, test_groups = train_test_split(
        temp_df[group_col].to_numpy(),
        test_size=relative_test,
        random_state=seed,
        stratify=temp_labels if can_stratify_temp else None,
    )

    split_map = {group: "train" for group in train_groups}
    split_map.update({group: "val" for group in val_groups})
    split_map.update({group: "test" for group in test_groups})

    split_df = df.copy()
    split_df["split"] = split_df[group_col].map(split_map)
    return split_df
