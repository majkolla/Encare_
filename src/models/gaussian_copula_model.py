from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from src.data.mixed import MixedColumnEncoding, restore_mixed_column, split_mixed_column
from src.models.base import BaseSynthesizer
from src.utils.types import Schema

_EPSILON = 1e-6
_MISSING_TOKEN = "__MISSING__"


@dataclass
class _NumericState:
    missing_rate: float
    sorted_values: np.ndarray
    probabilities: np.ndarray
    is_integer: bool
    support_values: np.ndarray


@dataclass
class _CategoricalState:
    categories: list[Any]
    probabilities: np.ndarray


@dataclass
class _ConstantState:
    value: Any
    missing_rate: float


class GaussianCopulaSynthesizer(BaseSynthesizer):
    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.schema: Schema | None = None
        self.config: dict[str, Any] = {}
        self.column_states: dict[str, Any] = {}
        self.correlation_matrix: np.ndarray | None = None
        self.modeled_columns: list[str] = []
        self.mixed_column_states: dict[str, MixedColumnEncoding] = {}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        normalized_states: dict[str, Any] = {}
        normalized_mixed_states: dict[str, MixedColumnEncoding] = {}

        for column_name, column_state in getattr(self, "column_states", {}).items():
            if isinstance(column_state, _ConstantState):
                normalized_states[column_name] = _ConstantState(
                    value=column_state.value,
                    missing_rate=getattr(column_state, "missing_rate", 0.0),
                )
                continue

            if isinstance(column_state, _NumericState):
                normalized_states[column_name] = _NumericState(
                    missing_rate=column_state.missing_rate,
                    sorted_values=column_state.sorted_values,
                    probabilities=column_state.probabilities,
                    is_integer=column_state.is_integer,
                    support_values=getattr(column_state, "support_values", np.array([], dtype=float)),
                )
                continue

            normalized_states[column_name] = column_state

        for column_name, column_state in getattr(self, "mixed_column_states", {}).items():
            if isinstance(column_state, MixedColumnEncoding):
                normalized_mixed_states[column_name] = column_state
                continue
            if isinstance(column_state, dict):
                normalized_mixed_states[column_name] = MixedColumnEncoding(
                    name=str(column_state.get("name", column_name)),
                    source_kind=str(column_state.get("source_kind", "numeric")),
                    value_column=str(column_state.get("value_column", f"{column_name}__value")),
                    state_column=str(column_state.get("state_column", f"{column_name}__state")),
                    numeric_support=np.asarray(column_state.get("numeric_support", []), dtype=float),
                    numeric_support_strings=list(column_state.get("numeric_support_strings", [])),
                )

        self.column_states = normalized_states
        self.mixed_column_states = normalized_mixed_states

    def fit(self, df: pd.DataFrame, schema: Schema, config: dict[str, Any]) -> "GaussianCopulaSynthesizer":
        self.schema = schema
        self.config = config
        self.modeled_columns = []
        latent_columns: list[np.ndarray] = []
        self.column_states = {}
        self.mixed_column_states = {}
        mixed_column_strategy = str(config.get("mixed_column_strategy", "split")).lower()

        for column in schema.columns:
            column_name = column.name
            series = df[column_name]

            if column.kind == "constant":
                non_null = series.dropna()
                value = non_null.iloc[0] if not non_null.empty else np.nan
                self.column_states[column_name] = _ConstantState(
                    value=value,
                    missing_rate=float(series.isna().mean()),
                )
                continue

        for column in schema.columns:
            column_name = column.name
            if column.kind == "constant":
                continue
            series = df[column_name]

            if column.mixed_value_kind is not None and mixed_column_strategy == "split":
                mixed_state, value_series, state_series = split_mixed_column(
                    column_name=column_name,
                    series=series,
                    source_kind=column.mixed_value_kind,
                    snap_numeric_max_unique=int(config.get("snap_numeric_max_unique", 32)),
                )
                value_state, value_latent = self._fit_numeric(
                    value_series,
                    int(pd.to_numeric(value_series, errors="coerce").dropna().nunique()),
                )
                state_state, state_latent = self._fit_categorical(state_series)
                self.mixed_column_states[column_name] = mixed_state
                self.column_states[mixed_state.value_column] = value_state
                self.column_states[mixed_state.state_column] = state_state
                self.modeled_columns.extend([mixed_state.value_column, mixed_state.state_column])
                latent_columns.extend([value_latent, state_latent])
                continue

            if column.kind == "numeric":
                state, latent = self._fit_numeric(series, column.unique_count)
            elif column.kind in {"categorical", "binary", "id_like", "datetime", "time"}:
                state, latent = self._fit_categorical(series)
            else:
                non_null = series.dropna()
                state = _ConstantState(
                    value=non_null.iloc[0] if not non_null.empty else np.nan,
                    missing_rate=float(series.isna().mean()),
                )
                latent = np.zeros(len(series))

            self.column_states[column_name] = state
            self.modeled_columns.append(column_name)
            latent_columns.append(latent)

        latent_matrix = np.column_stack(latent_columns) if latent_columns else np.zeros((len(df), 0))
        self.correlation_matrix = _regularize_correlation(np.corrcoef(latent_matrix, rowvar=False))
        return self

    def sample(self, n_rows: int) -> pd.DataFrame:
        if self.schema is None or self.correlation_matrix is None:
            raise RuntimeError("Model must be fit before sampling.")

        latent = self.rng.multivariate_normal(
            mean=np.zeros(len(self.modeled_columns)),
            cov=self.correlation_matrix,
            size=n_rows,
        )

        sampled_feature_columns: dict[str, pd.Series] = {}
        for index, column_name in enumerate(self.modeled_columns):
            state = self.column_states[column_name]
            u = norm.cdf(latent[:, index])

            if isinstance(state, _NumericState):
                sampled_feature_columns[column_name] = self._sample_numeric(state, u)
            else:
                sampled_feature_columns[column_name] = self._sample_categorical(state, u)

        sampled_columns: dict[str, pd.Series] = {}
        for column in self.schema.columns:
            if column.name in self.mixed_column_states:
                mixed_state = self.mixed_column_states[column.name]
                sampled_columns[column.name] = restore_mixed_column(
                    pd.DataFrame(sampled_feature_columns, index=range(n_rows)),
                    mixed_state,
                )
                continue

            if column.name in sampled_feature_columns:
                sampled_columns[column.name] = sampled_feature_columns[column.name]
                continue

            state = self.column_states.get(column.name)
            if isinstance(state, _ConstantState):
                sampled_columns[column.name] = self._sample_constant(state, n_rows)
            else:
                sampled_columns[column.name] = pd.Series(np.nan, index=range(n_rows))

        sampled = pd.DataFrame(sampled_columns, index=range(n_rows))
        return sampled[self.schema.column_order]

    def sample_and_inverse_transform(self, n_rows: int) -> pd.DataFrame:
        return self.sample(n_rows)

    def _fit_numeric(self, series: pd.Series, unique_count: int) -> tuple[_NumericState, np.ndarray]:
        numeric = pd.to_numeric(series, errors="coerce")
        observed = numeric.dropna().to_numpy(dtype=float)

        if len(observed) == 0:
            state = _NumericState(
                missing_rate=1.0,
                sorted_values=np.array([], dtype=float),
                probabilities=np.array([], dtype=float),
                is_integer=False,
                support_values=np.array([], dtype=float),
            )
            return state, np.zeros(len(series))

        ranks = rankdata(observed, method="average")
        observed_probabilities = ranks / (len(observed) + 1.0)
        missing_rate = float(numeric.isna().mean())

        latent_probabilities = np.full(len(series), fill_value=np.clip(missing_rate / 2.0, _EPSILON, 1.0 - _EPSILON))
        latent_probabilities[~numeric.isna()] = np.clip(
            missing_rate + (1.0 - missing_rate) * observed_probabilities,
            _EPSILON,
            1.0 - _EPSILON,
        )

        sorted_values = np.sort(observed)
        probabilities = np.linspace(1.0 / len(sorted_values), 1.0, len(sorted_values))
        state = _NumericState(
            missing_rate=missing_rate,
            sorted_values=sorted_values,
            probabilities=probabilities,
            is_integer=bool(np.allclose(observed, np.round(observed))),
            support_values=_numeric_support_values(observed, unique_count, self.config),
        )
        return state, norm.ppf(latent_probabilities)

    def _fit_categorical(self, series: pd.Series) -> tuple[_CategoricalState, np.ndarray]:
        filled = series.astype("object").where(series.notna(), _MISSING_TOKEN)
        counts = filled.value_counts(normalize=True, sort=False)
        categories = counts.index.tolist()
        probabilities = counts.to_numpy(dtype=float)
        cumulative = np.cumsum(probabilities)
        lower = cumulative - probabilities
        midpoints = np.clip(lower + probabilities / 2.0, _EPSILON, 1.0 - _EPSILON)
        midpoint_map = dict(zip(categories, midpoints))
        latent = filled.map(midpoint_map).to_numpy(dtype=float)
        state = _CategoricalState(categories=categories, probabilities=probabilities)
        return state, norm.ppf(latent)

    def _sample_numeric(self, state: _NumericState, u: np.ndarray) -> pd.Series:
        if len(state.sorted_values) == 0:
            return pd.Series([np.nan] * len(u))

        missing_mask = u < state.missing_rate
        adjusted_u = np.clip((u - state.missing_rate) / max(1.0 - state.missing_rate, _EPSILON), 0.0, 1.0)
        values = np.interp(adjusted_u, state.probabilities, state.sorted_values)

        if len(state.support_values) > 0:
            values = _snap_to_support(values, state.support_values)

        if state.is_integer:
            values = np.round(values)

        values = values.astype(float)
        values[missing_mask] = np.nan
        return pd.Series(values)

    def _sample_categorical(self, state: _CategoricalState, u: np.ndarray) -> pd.Series:
        cumulative = np.cumsum(state.probabilities)
        indices = np.searchsorted(cumulative, np.clip(u, 0.0, 1.0), side="right")
        indices = np.clip(indices, 0, len(state.categories) - 1)
        values = np.array([state.categories[idx] for idx in indices], dtype="object")
        values[values == _MISSING_TOKEN] = np.nan
        return pd.Series(values, dtype="object")

    def _sample_constant(self, state: _ConstantState, n_rows: int) -> pd.Series:
        values = pd.Series([state.value] * n_rows, dtype="object")
        if state.missing_rate <= 0.0:
            return values

        missing_count = int(round(state.missing_rate * n_rows))
        if missing_count <= 0:
            return values

        missing_count = min(missing_count, n_rows)
        missing_positions = self.rng.choice(n_rows, size=missing_count, replace=False)
        values.iloc[missing_positions] = np.nan
        return values


def _regularize_correlation(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim == 0:
        return np.ones((1, 1))
    if matrix.ndim == 1:
        return np.eye(len(matrix))
    if matrix.size == 0:
        return np.zeros((0, 0))

    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 1.0)

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.clip(eigenvalues, 1e-4, None)
    regularized = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    scaling = np.sqrt(np.diag(regularized))
    regularized = regularized / np.outer(scaling, scaling)
    np.fill_diagonal(regularized, 1.0)
    return regularized


def _numeric_support_values(
    observed: np.ndarray,
    unique_count: int,
    config: dict[str, Any],
) -> np.ndarray:
    max_unique = int(config.get("snap_numeric_max_unique", 32))
    if unique_count > max_unique:
        return np.array([], dtype=float)

    support_values = np.unique(observed.astype(float))
    if len(support_values) == 0 or len(support_values) > max_unique:
        return np.array([], dtype=float)

    return np.sort(support_values)


def _snap_to_support(values: np.ndarray, support_values: np.ndarray) -> np.ndarray:
    if len(support_values) == 0 or len(values) == 0:
        return values

    snapped = values.astype(float, copy=True)
    insert_positions = np.searchsorted(support_values, snapped, side="left")
    insert_positions = np.clip(insert_positions, 0, len(support_values) - 1)
    left_positions = np.clip(insert_positions - 1, 0, len(support_values) - 1)
    right_positions = insert_positions
    left_values = support_values[left_positions]
    right_values = support_values[right_positions]
    choose_left = np.abs(snapped - left_values) <= np.abs(snapped - right_values)
    return np.where(choose_left, left_values, right_values)
