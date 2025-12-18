import pandas as pd

from trustquerynet.data.splits import make_group_stratified_split


def test_group_split_has_no_leakage():
    df = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(12)],
            "group_id": ["g0", "g0", "g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4", "g5", "g5"],
            "y_clean": [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2],
        }
    )
    split_df = make_group_stratified_split(
        df=df,
        label_col="y_clean",
        group_col="group_id",
        seed=42,
        ratios={"train": 0.5, "val": 0.25, "test": 0.25},
    )
    per_group = split_df.groupby("group_id")["split"].nunique()
    assert per_group.max() == 1
