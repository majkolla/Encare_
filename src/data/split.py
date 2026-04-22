from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.model_selection import KFold, train_test_split


def make_train_val_split(
    df: pd.DataFrame,
    seed: int,
    val_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, val_df = train_test_split(
        df,
        test_size=val_frac,
        random_state=seed,
        shuffle=True,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def make_k_splits(
    df: pd.DataFrame,
    seeds: Iterable[int],
    n_splits: int = 5,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=min(seeds, default=42))
    splits: list[tuple[pd.DataFrame, pd.DataFrame]] = []

    for train_idx, val_idx in splitter.split(df):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        splits.append((train_df, val_df))

    return splits

