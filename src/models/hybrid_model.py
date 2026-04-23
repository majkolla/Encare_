from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.eval.score import compute_total_score
from src.models.base import BaseSynthesizer
from src.models.ctgan_model import CTGANSynthesizer
from src.models.gaussian_copula_model import GaussianCopulaSynthesizer
from src.rules.repair import repair_dataframe
from src.utils.types import Schema


class HybridSynthesizer(BaseSynthesizer):
    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.schema: Schema | None = None
        self.config: dict[str, Any] = {}
        self.alpha: float = 0.5
        self.copula_model = GaussianCopulaSynthesizer(seed=seed)
        self.ctgan_model = CTGANSynthesizer(seed=seed)

    def fit(self, df: pd.DataFrame, schema: Schema, config: dict[str, Any]) -> "HybridSynthesizer":
        self.schema = schema
        self.config = config
        self.alpha = float(config.get("alpha", 0.5))

        self.copula_model.fit(df, schema, config.get("copula", config))
        self.ctgan_model.fit(df, schema, config.get("ctgan", config))
        return self

    def sample(self, n_rows: int, alpha: float | None = None) -> pd.DataFrame:
        if self.schema is None:
            raise RuntimeError("Model must be fit before sampling.")

        mixture_weight = self.alpha if alpha is None else alpha
        n_copula = int(round(mixture_weight * n_rows))
        n_ctgan = n_rows - n_copula

        frames = []
        if n_copula > 0:
            frames.append(self.copula_model.sample(n_copula))
        if n_ctgan > 0:
            frames.append(self.ctgan_model.sample(n_ctgan))

        synthetic = pd.concat(frames, ignore_index=True)
        synthetic = synthetic.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)
        return synthetic[self.schema.column_order]

    def grid_search_alpha(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        schema: Schema,
        constraints: dict,
        weights: dict[str, float],
        alphas: list[float] | None = None,
    ) -> tuple[float, list[dict[str, Any]]]:
        alpha_grid = alphas or list(self.config.get("alphas", [0.0, 0.25, 0.5, 0.75, 1.0]))
        results: list[dict[str, Any]] = []

        for alpha in alpha_grid:
            synthetic = self.sample(len(val_df), alpha=alpha)
            synthetic = repair_dataframe(synthetic, constraints)
            metrics = compute_total_score(val_df, synthetic, schema, constraints, weights)
            results.append({"alpha": alpha, "metrics": metrics})

        best = max(results, key=lambda item: item["metrics"]["total_score"])
        self.alpha = float(best["alpha"])
        return self.alpha, results
