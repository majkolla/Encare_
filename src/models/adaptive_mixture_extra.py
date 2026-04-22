from __future__ import annotations

from typing import Any

import pandas as pd

from src.models.hybrid_model import HybridSynthesizer
from src.utils.types import Schema


class AdaptiveMixtureExtraSynthesizer(HybridSynthesizer):
    def __init__(self, seed: int = 42) -> None:
        super().__init__(seed=seed)
        self.alpha_by_group: dict[Any, float] = {}
        self.group_column: str | None = None

    def fit(self, df: pd.DataFrame, schema: Schema, config: dict[str, Any]) -> "AdaptiveMixtureExtraSynthesizer":
        super().fit(df, schema, config)
        self.group_column = config.get("group_column")
        if self.group_column and self.group_column in df.columns:
            grouped = df[self.group_column].fillna("__MISSING__")
            for value in grouped.drop_duplicates():
                self.alpha_by_group[value] = float(config.get("alpha", 0.5))
        return self

    def sample_adaptive(self, n_rows: int, subgroup_plan: dict[Any, int], alpha_by_group: dict[Any, float]) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for group, count in subgroup_plan.items():
            alpha = alpha_by_group.get(group, self.alpha)
            frames.append(self.sample(count, alpha=alpha))
        return pd.concat(frames, ignore_index=True)

