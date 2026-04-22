from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.models.base import BaseSynthesizer
from src.utils.types import Schema


@dataclass
class _BaselineColumnState:
    kind: str
    values: np.ndarray
    missing_rate: float
    min_value: float | None = None
    max_value: float | None = None
    is_integer: bool = False


class IndependentBaselineSynthesizer(BaseSynthesizer):
    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.schema: Schema | None = None
        self.config: dict[str, Any] = {}
        self.column_states: dict[str, _BaselineColumnState] = {}

    def fit(self, df: pd.DataFrame, schema: Schema, config: dict[str, Any]) -> "IndependentBaselineSynthesizer":
        self.schema = schema
        self.config = config
        self.column_states = {}

        for column in schema.columns:
            series = df[column.name]
            missing_rate = float(series.isna().mean())

            if column.kind == "numeric":
                numeric = pd.to_numeric(series, errors="coerce").dropna()
                values = numeric.to_numpy(dtype=float)
                self.column_states[column.name] = _BaselineColumnState(
                    kind=column.kind,
                    values=values,
                    missing_rate=missing_rate,
                    min_value=float(numeric.min()) if not numeric.empty else None,
                    max_value=float(numeric.max()) if not numeric.empty else None,
                    is_integer=bool(not numeric.empty and np.allclose(values, np.round(values))),
                )
                continue

            values = series.dropna().astype("object").to_numpy()
            self.column_states[column.name] = _BaselineColumnState(
                kind=column.kind,
                values=values,
                missing_rate=missing_rate,
            )

        return self

    def sample(self, n_rows: int) -> pd.DataFrame:
        if self.schema is None:
            raise RuntimeError("Model must be fit before sampling.")

        numeric_strategy = self.config.get("numeric_strategy", "bootstrap")
        output = pd.DataFrame(index=range(n_rows))

        for column in self.schema.columns:
            state = self.column_states[column.name]
            if state.kind == "numeric":
                output[column.name] = self._sample_numeric(state, n_rows, numeric_strategy)
            else:
                output[column.name] = self._sample_discrete(state, n_rows)

        return output[self.schema.column_order]

    def _sample_numeric(
        self,
        state: _BaselineColumnState,
        n_rows: int,
        strategy: str,
    ) -> pd.Series:
        if len(state.values) == 0:
            return pd.Series([np.nan] * n_rows)

        if strategy == "uniform" and state.min_value is not None and state.max_value is not None:
            samples = self.rng.uniform(state.min_value, state.max_value, n_rows)
        else:
            samples = self.rng.choice(state.values, size=n_rows, replace=True)

        if state.is_integer:
            samples = np.round(samples)

        missing_mask = self.rng.random(n_rows) < state.missing_rate
        samples = samples.astype(float)
        samples[missing_mask] = np.nan
        return pd.Series(samples)

    def _sample_discrete(self, state: _BaselineColumnState, n_rows: int) -> pd.Series:
        if len(state.values) == 0:
            return pd.Series([np.nan] * n_rows, dtype="object")

        samples = self.rng.choice(state.values, size=n_rows, replace=True).astype("object")
        missing_mask = self.rng.random(n_rows) < state.missing_rate
        samples[missing_mask] = np.nan
        return pd.Series(samples, dtype="object")

