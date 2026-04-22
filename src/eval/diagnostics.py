from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.data.preprocess import fit_preprocessor, transform_for_model
from src.eval.dependencies import dependency_pair_drift
from src.eval.discriminator import compute_discriminator_diagnostics
from src.eval.marginals import (
    jsd_by_categorical_column,
    ks_by_numeric_column,
    tv_by_categorical_column,
    wasserstein_by_numeric_column,
)
from src.utils.types import ColumnSchema, Schema

_AGE_PATTERN = re.compile(r"age", re.IGNORECASE)
_GENDER_PATTERN = re.compile(r"(gender|sex)", re.IGNORECASE)
_SURGERY_PATTERN = re.compile(r"(procedure|surgery|operation)", re.IGNORECASE)
_LOS_PATTERN = re.compile(r"length of stay", re.IGNORECASE)


def compute_run_diagnostics(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    schema: Schema,
    limit_numeric: int = 25,
    limit_categorical: int = 25,
    limit_pairs: int = 50,
) -> dict[str, Any]:
    numeric_scores = _rank_metric_scores(
        {
            "ks": ks_by_numeric_column(real_df, syn_df, schema),
            "wasserstein": wasserstein_by_numeric_column(real_df, syn_df, schema),
        },
        limit=limit_numeric,
    )
    categorical_scores = _rank_metric_scores(
        {
            "tv": tv_by_categorical_column(real_df, syn_df, schema),
            "jsd": jsd_by_categorical_column(real_df, syn_df, schema),
        },
        limit=limit_categorical,
    )
    subgroup_specs = _build_subgroup_specs(real_df, syn_df, schema)

    return {
        "worst_numeric_columns": numeric_scores,
        "worst_categorical_columns": categorical_scores,
        "worst_dependency_pairs": dependency_pair_drift(real_df, syn_df, schema, limit=limit_pairs),
        "subgroup_drift": [
            _summarize_subgroup_drift(spec, limit=8)
            for spec in subgroup_specs
        ],
        "discriminator": compute_discriminator_diagnostics(real_df, syn_df, schema),
        "privacy_risk_by_subgroup": _privacy_risk_by_subgroup(real_df, syn_df, schema, subgroup_specs),
    }


def _rank_metric_scores(
    metric_scores: dict[str, dict[str, float]],
    limit: int,
) -> list[dict[str, Any]]:
    aggregated: dict[str, dict[str, Any]] = {}

    for metric_name, scores in metric_scores.items():
        for column_name, score in scores.items():
            entry = aggregated.setdefault(
                column_name,
                {"column": column_name, "score": 0.0, "metrics": {}},
            )
            entry["metrics"][metric_name] = float(score)

    for entry in aggregated.values():
        metric_values = list(entry["metrics"].values())
        entry["score"] = float(np.mean(metric_values)) if metric_values else 0.0

    ranked = sorted(aggregated.values(), key=lambda item: item["score"])
    return ranked[:limit]


def _build_subgroup_specs(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    schema: Schema,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []

    age_column = _select_column(schema.columns, _AGE_PATTERN, kind_group="numeric_like")
    if age_column is not None:
        specs.append(
            _bucket_numeric_subgroup(
                name="age_band",
                column=age_column,
                real_df=real_df,
                syn_df=syn_df,
                bins=[0, 50, 65, 75, 85, 120, np.inf],
                labels=["<50", "50-64", "65-74", "75-84", "85-120", "120+"],
            )
        )

    gender_column = _select_column(schema.columns, _GENDER_PATTERN, kind_group="categorical")
    if gender_column is not None:
        specs.append(
            _categorical_subgroup(
                name="sex_or_gender",
                column=gender_column,
                real_df=real_df,
                syn_df=syn_df,
            )
        )

    surgery_column = _select_column(schema.columns, _SURGERY_PATTERN, kind_group="categorical", max_unique=40)
    if surgery_column is not None:
        specs.append(
            _categorical_subgroup(
                name="surgery_type",
                column=surgery_column,
                real_df=real_df,
                syn_df=syn_df,
                max_levels=12,
            )
        )

    los_column = _select_column(schema.columns, _LOS_PATTERN, kind_group="numeric_like")
    if los_column is not None:
        specs.append(
            _quantile_numeric_subgroup(
                name="length_of_stay_bucket",
                column=los_column,
                real_df=real_df,
                syn_df=syn_df,
                quantiles=[0.0, 0.25, 0.5, 0.75, 1.0],
            )
        )

    specs.append(
        {
            "name": "missingness_pattern",
            "column": "__row_missing_count__",
            "real_labels": _bucket_missingness(real_df),
            "syn_labels": _bucket_missingness(syn_df),
        }
    )
    return [spec for spec in specs if spec["real_labels"].notna().any() or spec["syn_labels"].notna().any()]


def _summarize_subgroup_drift(
    spec: dict[str, Any],
    limit: int,
) -> dict[str, Any]:
    real_counts = spec["real_labels"].astype("object").fillna("__MISSING__").value_counts(normalize=True)
    syn_counts = spec["syn_labels"].astype("object").fillna("__MISSING__").value_counts(normalize=True)
    levels = real_counts.index.union(syn_counts.index)
    rows = []
    for level in levels:
        real_rate = float(real_counts.get(level, 0.0))
        syn_rate = float(syn_counts.get(level, 0.0))
        rows.append(
            {
                "value": str(level),
                "real_rate": real_rate,
                "synthetic_rate": syn_rate,
                "abs_diff": abs(real_rate - syn_rate),
            }
        )
    ranked = sorted(rows, key=lambda item: item["abs_diff"], reverse=True)
    return {
        "name": spec["name"],
        "column": spec["column"],
        "top_shifts": ranked[:limit],
    }


def _privacy_risk_by_subgroup(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    schema: Schema,
    subgroup_specs: list[dict[str, Any]],
    max_rows: int = 2000,
) -> list[dict[str, Any]]:
    if real_df.empty or syn_df.empty:
        return []

    real_sample = real_df.sample(min(len(real_df), max_rows), random_state=42)
    syn_sample = syn_df.sample(min(len(syn_df), max_rows), random_state=42)
    combined = pd.concat(
        [real_sample.reset_index(drop=True), syn_sample.reset_index(drop=True)],
        axis=0,
        ignore_index=True,
    )
    preprocessor = fit_preprocessor(combined, schema)
    encoded = transform_for_model(combined, preprocessor, model_name="privacy")
    encoded_real = encoded.iloc[: len(real_sample)].reset_index(drop=True)
    encoded_syn = encoded.iloc[len(real_sample) :].reset_index(drop=True)

    if encoded_real.empty or encoded_syn.empty:
        return []

    model = NearestNeighbors(n_neighbors=1, metric="euclidean")
    model.fit(encoded_real)
    distances, _ = model.kneighbors(encoded_syn)
    syn_distances = pd.Series(distances[:, 0], index=syn_sample.index)

    summaries: list[dict[str, Any]] = []
    for spec in subgroup_specs:
        syn_labels = spec["syn_labels"]
        if len(syn_labels) != len(syn_df):
            continue
        sampled_labels = syn_labels.loc[syn_sample.index]
        for value, index in sampled_labels.groupby(sampled_labels.astype("object").fillna("__MISSING__")).groups.items():
            group_distances = syn_distances.loc[list(index)]
            if len(group_distances) < 5:
                continue
            summaries.append(
                {
                    "group": spec["name"],
                    "value": str(value),
                    "rows": int(len(group_distances)),
                    "median_distance": float(group_distances.median()),
                    "p05_distance": float(group_distances.quantile(0.05)),
                }
            )

    ranked = sorted(
        summaries,
        key=lambda item: (item["median_distance"], item["p05_distance"]),
    )
    return ranked[:12]


def _categorical_subgroup(
    name: str,
    column: ColumnSchema,
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    max_levels: int = 10,
) -> dict[str, Any]:
    real_series = real_df[column.name].astype("object")
    syn_series = syn_df[column.name].astype("object")
    top_levels = (
        real_series.fillna("__MISSING__")
        .value_counts()
        .head(max_levels)
        .index
        .tolist()
    )
    real_labels = real_series.where(real_series.isin(top_levels), "__OTHER__")
    syn_labels = syn_series.where(syn_series.isin(top_levels), "__OTHER__")
    return {
        "name": name,
        "column": column.name,
        "real_labels": real_labels,
        "syn_labels": syn_labels,
    }


def _bucket_numeric_subgroup(
    name: str,
    column: ColumnSchema,
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    bins: list[float],
    labels: list[str],
) -> dict[str, Any]:
    real_numeric = pd.to_numeric(real_df[column.name], errors="coerce")
    syn_numeric = pd.to_numeric(syn_df[column.name], errors="coerce")
    return {
        "name": name,
        "column": column.name,
        "real_labels": pd.cut(real_numeric, bins=bins, labels=labels, include_lowest=True),
        "syn_labels": pd.cut(syn_numeric, bins=bins, labels=labels, include_lowest=True),
    }


def _quantile_numeric_subgroup(
    name: str,
    column: ColumnSchema,
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    quantiles: list[float],
) -> dict[str, Any]:
    real_numeric = pd.to_numeric(real_df[column.name], errors="coerce")
    syn_numeric = pd.to_numeric(syn_df[column.name], errors="coerce")
    valid_real = real_numeric.dropna()
    if valid_real.empty:
        labels = pd.Series(np.nan, index=real_numeric.index)
        syn_labels = pd.Series(np.nan, index=syn_numeric.index)
        return {"name": name, "column": column.name, "real_labels": labels, "syn_labels": syn_labels}

    edges = valid_real.quantile(quantiles).to_numpy(dtype=float)
    edges = np.unique(edges)
    if len(edges) < 2:
        edges = np.array([float(valid_real.min()), float(valid_real.max()) + 1.0])
    edges[0] = -np.inf
    edges[-1] = np.inf
    bucket_labels = [f"q{index + 1}" for index in range(len(edges) - 1)]
    return {
        "name": name,
        "column": column.name,
        "real_labels": pd.cut(real_numeric, bins=edges, labels=bucket_labels, include_lowest=True),
        "syn_labels": pd.cut(syn_numeric, bins=edges, labels=bucket_labels, include_lowest=True),
    }


def _bucket_missingness(df: pd.DataFrame) -> pd.Series:
    missing_counts = df.isna().sum(axis=1)
    return pd.cut(
        missing_counts,
        bins=[-1, 5, 15, 30, 60, np.inf],
        labels=["0-5", "6-15", "16-30", "31-60", "60+"],
    )


def _select_column(
    columns: list[ColumnSchema],
    pattern: re.Pattern[str],
    kind_group: str,
    max_unique: int | None = None,
) -> ColumnSchema | None:
    candidates: list[ColumnSchema] = []
    for column in columns:
        if not pattern.search(column.name):
            continue
        if max_unique is not None and column.unique_count > max_unique:
            continue
        if kind_group == "categorical" and column.kind not in {"categorical", "binary", "id_like"}:
            continue
        if kind_group == "numeric_like" and column.kind != "numeric" and column.mixed_value_kind != "numeric":
            continue
        candidates.append(column)

    if not candidates:
        return None
    return sorted(candidates, key=lambda column: (column.unique_count, len(column.name)))[0]
