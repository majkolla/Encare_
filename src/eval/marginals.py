from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from src.data.mixed import split_mixed_column
from src.utils.types import Schema


def ks_by_numeric_column(
    real: pd.DataFrame,
    syn: pd.DataFrame,
    schema: Schema,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for column in schema.numeric_columns:
        real_values = pd.to_numeric(real[column], errors="coerce").dropna()
        syn_values = pd.to_numeric(syn[column], errors="coerce").dropna()
        if len(real_values) == 0 or len(syn_values) == 0:
            continue
        statistic, _ = stats.ks_2samp(real_values, syn_values)
        scores[column] = float(1.0 - statistic)

    for column_schema in schema.columns:
        if column_schema.mixed_value_kind is None:
            continue
        feature_name = f"{column_schema.name}__value"
        _, real_values, _ = split_mixed_column(
            column_name=column_schema.name,
            series=real[column_schema.name],
            source_kind=column_schema.mixed_value_kind,
        )
        _, syn_values, _ = split_mixed_column(
            column_name=column_schema.name,
            series=syn[column_schema.name],
            source_kind=column_schema.mixed_value_kind,
        )
        real_values = pd.to_numeric(real_values, errors="coerce").dropna()
        syn_values = pd.to_numeric(syn_values, errors="coerce").dropna()
        if len(real_values) == 0 or len(syn_values) == 0:
            continue
        statistic, _ = stats.ks_2samp(real_values, syn_values)
        scores[feature_name] = float(1.0 - statistic)
    return scores


def wasserstein_by_numeric_column(
    real: pd.DataFrame,
    syn: pd.DataFrame,
    schema: Schema,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for column in schema.numeric_columns:
        real_values = pd.to_numeric(real[column], errors="coerce").dropna()
        syn_values = pd.to_numeric(syn[column], errors="coerce").dropna()
        if len(real_values) == 0 or len(syn_values) == 0:
            continue
        distance = stats.wasserstein_distance(real_values, syn_values)
        scale = float(real_values.std(ddof=0) or 1.0)
        scores[column] = float(1.0 / (1.0 + distance / scale))

    for column_schema in schema.columns:
        if column_schema.mixed_value_kind is None:
            continue
        feature_name = f"{column_schema.name}__value"
        _, real_values, _ = split_mixed_column(
            column_name=column_schema.name,
            series=real[column_schema.name],
            source_kind=column_schema.mixed_value_kind,
        )
        _, syn_values, _ = split_mixed_column(
            column_name=column_schema.name,
            series=syn[column_schema.name],
            source_kind=column_schema.mixed_value_kind,
        )
        real_values = pd.to_numeric(real_values, errors="coerce").dropna()
        syn_values = pd.to_numeric(syn_values, errors="coerce").dropna()
        if len(real_values) == 0 or len(syn_values) == 0:
            continue
        distance = stats.wasserstein_distance(real_values, syn_values)
        scale = float(real_values.std(ddof=0) or 1.0)
        scores[feature_name] = float(1.0 / (1.0 + distance / scale))
    return scores


def tv_by_categorical_column(
    real: pd.DataFrame,
    syn: pd.DataFrame,
    schema: Schema,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for column in schema.categorical_columns:
        real_dist, syn_dist = _aligned_categorical_distributions(real[column], syn[column])
        scores[column] = float(1.0 - 0.5 * np.abs(real_dist - syn_dist).sum())

    for column_schema in schema.columns:
        if column_schema.mixed_value_kind is None:
            continue
        feature_name = f"{column_schema.name}__state"
        _, _, real_states = split_mixed_column(
            column_name=column_schema.name,
            series=real[column_schema.name],
            source_kind=column_schema.mixed_value_kind,
        )
        _, _, syn_states = split_mixed_column(
            column_name=column_schema.name,
            series=syn[column_schema.name],
            source_kind=column_schema.mixed_value_kind,
        )
        real_dist, syn_dist = _aligned_categorical_distributions(real_states, syn_states)
        scores[feature_name] = float(1.0 - 0.5 * np.abs(real_dist - syn_dist).sum())
    return scores


def jsd_by_categorical_column(
    real: pd.DataFrame,
    syn: pd.DataFrame,
    schema: Schema,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for column in schema.categorical_columns:
        real_dist, syn_dist = _aligned_categorical_distributions(real[column], syn[column])
        midpoint = 0.5 * (real_dist + syn_dist)
        jsd = 0.5 * (_safe_entropy(real_dist, midpoint) + _safe_entropy(syn_dist, midpoint))
        scores[column] = float(1.0 - jsd / math.log(2))

    for column_schema in schema.columns:
        if column_schema.mixed_value_kind is None:
            continue
        feature_name = f"{column_schema.name}__state"
        _, _, real_states = split_mixed_column(
            column_name=column_schema.name,
            series=real[column_schema.name],
            source_kind=column_schema.mixed_value_kind,
        )
        _, _, syn_states = split_mixed_column(
            column_name=column_schema.name,
            series=syn[column_schema.name],
            source_kind=column_schema.mixed_value_kind,
        )
        real_dist, syn_dist = _aligned_categorical_distributions(real_states, syn_states)
        midpoint = 0.5 * (real_dist + syn_dist)
        jsd = 0.5 * (_safe_entropy(real_dist, midpoint) + _safe_entropy(syn_dist, midpoint))
        scores[feature_name] = float(1.0 - jsd / math.log(2))
    return scores


def aggregate_marginal_score(metrics_dict: dict[str, dict[str, float]]) -> dict[str, float | dict[str, float]]:
    metric_scores = [
        np.mean(list(metric.values()))
        for metric in metrics_dict.values()
        if metric
    ]
    aggregate = float(np.mean(metric_scores)) if metric_scores else 0.0
    return {"score": aggregate, "details": metrics_dict}


def _aligned_categorical_distributions(
    real_series: pd.Series,
    syn_series: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    real_counts = real_series.astype("object").fillna("__MISSING__").value_counts(normalize=True)
    syn_counts = syn_series.astype("object").fillna("__MISSING__").value_counts(normalize=True)
    index = real_counts.index.union(syn_counts.index)
    real_dist = real_counts.reindex(index, fill_value=0.0).to_numpy(dtype=float)
    syn_dist = syn_counts.reindex(index, fill_value=0.0).to_numpy(dtype=float)
    return real_dist, syn_dist


def _safe_entropy(p: np.ndarray, q: np.ndarray) -> float:
    mask = (p > 0) & (q > 0)
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))
