from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.mixed import datetime_to_numeric, split_mixed_column, time_to_numeric
from src.utils.types import Schema


def pearson_corr_diff(
    real_numeric: pd.DataFrame,
    syn_numeric: pd.DataFrame,
) -> dict[str, float]:
    if real_numeric.shape[1] < 2 or syn_numeric.shape[1] < 2:
        return {"score": 0.0, "mean_abs_diff": 0.0}

    real_corr = real_numeric.corr(method="pearson").fillna(0.0)
    syn_corr = syn_numeric.corr(method="pearson").fillna(0.0)
    mean_abs_diff = float(np.abs(real_corr - syn_corr).to_numpy().mean())
    return {"score": max(0.0, 1.0 - mean_abs_diff / 2.0), "mean_abs_diff": mean_abs_diff}


def spearman_corr_diff(
    real_numeric: pd.DataFrame,
    syn_numeric: pd.DataFrame,
) -> dict[str, float]:
    if real_numeric.shape[1] < 2 or syn_numeric.shape[1] < 2:
        return {"score": 0.0, "mean_abs_diff": 0.0}

    real_corr = real_numeric.corr(method="spearman").fillna(0.0)
    syn_corr = syn_numeric.corr(method="spearman").fillna(0.0)
    mean_abs_diff = float(np.abs(real_corr - syn_corr).to_numpy().mean())
    return {"score": max(0.0, 1.0 - mean_abs_diff / 2.0), "mean_abs_diff": mean_abs_diff}


def cramers_v_matrix(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    if not cat_cols:
        return pd.DataFrame()

    matrix = pd.DataFrame(np.eye(len(cat_cols)), index=cat_cols, columns=cat_cols)

    for i, left in enumerate(cat_cols):
        for right in cat_cols[i + 1 :]:
            contingency = pd.crosstab(
                df[left].astype("object").fillna("__MISSING__"),
                df[right].astype("object").fillna("__MISSING__"),
            )
            if contingency.empty:
                value = 0.0
            else:
                chi2 = _chi2_stat(contingency.to_numpy())
                n = contingency.to_numpy().sum()
                k = min(contingency.shape) - 1
                value = np.sqrt((chi2 / max(n, 1)) / max(k, 1)) if n > 0 else 0.0
            matrix.loc[left, right] = value
            matrix.loc[right, left] = value

    return matrix


def mixed_association_matrix(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    numeric_frame, categorical_frame = dependency_feature_frames(df, schema)
    encoded_columns: dict[str, pd.Series] = {}

    for column_name in numeric_frame.columns:
        encoded_columns[column_name] = pd.to_numeric(numeric_frame[column_name], errors="coerce")

    for column_name in categorical_frame.columns:
        encoded_columns[column_name] = pd.Series(
            pd.Categorical(
                categorical_frame[column_name].astype("object").fillna("__MISSING__")
            ).codes.astype(float),
            index=df.index,
        )

    encoded = pd.DataFrame(encoded_columns, index=df.index)
    return encoded.corr(method="spearman").fillna(0.0)


def dependency_feature_frames(
    df: pd.DataFrame,
    schema: Schema,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_columns: dict[str, pd.Series] = {}
    categorical_columns: dict[str, pd.Series] = {}

    for column in schema.columns:
        series = df[column.name]

        if column.mixed_value_kind is not None:
            encoding, value_series, state_series = split_mixed_column(
                column_name=column.name,
                series=series,
                source_kind=column.mixed_value_kind,
            )
            numeric_columns[encoding.value_column] = pd.to_numeric(value_series, errors="coerce")
            categorical_columns[encoding.state_column] = state_series.astype("object")
            continue

        if column.kind == "numeric":
            numeric_columns[column.name] = pd.to_numeric(series, errors="coerce")
            continue

        if column.kind == "datetime":
            numeric_columns[column.name] = datetime_to_numeric(series)
            continue

        if column.kind == "time":
            numeric_columns[column.name] = time_to_numeric(series)
            continue

        categorical_columns[column.name] = series.astype("object")

    return (
        pd.DataFrame(numeric_columns, index=df.index),
        pd.DataFrame(categorical_columns, index=df.index),
    )


def dependency_pair_drift(
    real: pd.DataFrame,
    syn: pd.DataFrame,
    schema: Schema,
    limit: int = 50,
) -> list[dict[str, float | str]]:
    numeric_real, categorical_real = dependency_feature_frames(real, schema)
    numeric_syn, categorical_syn = dependency_feature_frames(syn, schema)
    drift_rows: list[dict[str, float | str]] = []

    if numeric_real.shape[1] >= 2 and numeric_syn.shape[1] >= 2:
        for method in ("pearson", "spearman"):
            real_corr = numeric_real.corr(method=method).fillna(0.0)
            syn_corr = numeric_syn.corr(method=method).fillna(0.0)
            drift_rows.extend(
                _matrix_pair_rows(real_corr, syn_corr, metric=method)
            )

    cat_cols = list(categorical_real.columns)
    if cat_cols and list(categorical_syn.columns):
        shared_cat_cols = [column for column in cat_cols if column in categorical_syn.columns][:50]
        if len(shared_cat_cols) >= 2:
            real_cramers = cramers_v_matrix(categorical_real, shared_cat_cols)
            syn_cramers = cramers_v_matrix(categorical_syn, shared_cat_cols)
            drift_rows.extend(
                _matrix_pair_rows(real_cramers, syn_cramers, metric="cramers_v")
            )

    real_mixed = mixed_association_matrix(real, schema)
    syn_mixed = mixed_association_matrix(syn, schema)
    drift_rows.extend(_matrix_pair_rows(real_mixed, syn_mixed, metric="mixed_spearman"))

    ranked = sorted(drift_rows, key=lambda row: float(row["abs_diff"]), reverse=True)
    return ranked[:limit]


def dependency_score(real: pd.DataFrame, syn: pd.DataFrame, schema: Schema) -> dict[str, float | dict[str, float]]:
    real_numeric, real_categorical = dependency_feature_frames(real, schema)
    syn_numeric, syn_categorical = dependency_feature_frames(syn, schema)

    pearson = pearson_corr_diff(real_numeric, syn_numeric)
    spearman = spearman_corr_diff(real_numeric, syn_numeric)

    cat_cols = [column for column in real_categorical.columns if column in syn_categorical.columns][:50]
    if len(cat_cols) >= 2:
        real_cramers = cramers_v_matrix(real_categorical, cat_cols)
        syn_cramers = cramers_v_matrix(syn_categorical, cat_cols)
        cramers_diff = float(np.abs(real_cramers - syn_cramers).to_numpy().mean())
        cramers_score = max(0.0, 1.0 - cramers_diff)
    else:
        cramers_diff = 0.0
        cramers_score = 0.0

    real_mixed = mixed_association_matrix(real, schema)
    syn_mixed = mixed_association_matrix(syn, schema)
    mixed_diff = float(np.abs(real_mixed - syn_mixed).to_numpy().mean())
    mixed_score = max(0.0, 1.0 - mixed_diff / 2.0)

    aggregate = float(np.mean([pearson["score"], spearman["score"], cramers_score, mixed_score]))
    return {
        "score": aggregate,
        "details": {
            "pearson": pearson,
            "spearman": spearman,
            "cramers_v": {"score": cramers_score, "mean_abs_diff": cramers_diff},
            "mixed": {"score": mixed_score, "mean_abs_diff": mixed_diff},
        },
    }


def _matrix_pair_rows(
    real_matrix: pd.DataFrame,
    syn_matrix: pd.DataFrame,
    metric: str,
) -> list[dict[str, float | str]]:
    if real_matrix.empty or syn_matrix.empty:
        return []

    shared_index = [column for column in real_matrix.index if column in syn_matrix.index]
    if len(shared_index) < 2:
        return []

    real_aligned = real_matrix.loc[shared_index, shared_index]
    syn_aligned = syn_matrix.loc[shared_index, shared_index]
    rows: list[dict[str, float | str]] = []

    for left_index, left in enumerate(shared_index):
        for right in shared_index[left_index + 1 :]:
            real_value = float(real_aligned.loc[left, right])
            syn_value = float(syn_aligned.loc[left, right])
            rows.append(
                {
                    "metric": metric,
                    "left": left,
                    "right": right,
                    "real_value": real_value,
                    "synthetic_value": syn_value,
                    "abs_diff": abs(real_value - syn_value),
                }
            )

    return rows


def _chi2_stat(observed: np.ndarray) -> float:
    row_sums = observed.sum(axis=1, keepdims=True)
    col_sums = observed.sum(axis=0, keepdims=True)
    total = observed.sum()
    if total == 0:
        return 0.0
    expected = row_sums @ col_sums / total
    mask = expected > 0
    return float((((observed - expected) ** 2) / expected)[mask].sum())
