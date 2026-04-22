from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.data.preprocess import fit_preprocessor, transform_for_model
from src.utils.types import Schema


def encode_for_distance(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    preprocessor = fit_preprocessor(df, schema)
    encoded = transform_for_model(df, preprocessor, model_name="privacy")
    return _scale_distance_features(encoded)


def resolve_privacy_min_distance(
    real_df: pd.DataFrame,
    schema: Schema,
    min_distance: float = 0.0,
    min_distance_quantile: float | None = None,
    max_rows: int = 2000,
) -> dict[str, float]:
    resolved = float(min_distance)
    reference = {"resolved_min_distance": resolved}

    quantile = None if min_distance_quantile is None else float(min_distance_quantile)
    if quantile is None or quantile <= 0.0:
        return reference

    real_nn = compute_real_nearest_neighbor_distance(real_df, schema, max_rows=max_rows)
    quantile = float(np.clip(quantile, 0.0, 1.0))
    quantile_distance = float(np.quantile(real_nn["distances"], quantile))
    resolved = max(resolved, quantile_distance)
    return {
        "resolved_min_distance": resolved,
        "real_nn_median_distance": float(real_nn["median_distance"]),
        "real_nn_p05_distance": float(real_nn["p05_distance"]),
        "real_nn_quantile": quantile,
        "real_nn_quantile_distance": quantile_distance,
    }


def compute_real_nearest_neighbor_distance(
    real_df: pd.DataFrame,
    schema: Schema,
    max_rows: int = 2000,
) -> dict[str, float | np.ndarray]:
    real_sample = real_df.sample(min(len(real_df), max_rows), random_state=42).reset_index(drop=True)
    encoded_real = encode_for_distance(real_sample, schema)

    if encoded_real.empty:
        empty = np.array([], dtype=float)
        return {
            "distances": empty,
            "mean_distance": 0.0,
            "median_distance": 0.0,
            "p05_distance": 0.0,
        }

    if len(encoded_real) <= 1:
        zeros = np.zeros(len(encoded_real), dtype=float)
        return {
            "distances": zeros,
            "mean_distance": 0.0,
            "median_distance": 0.0,
            "p05_distance": 0.0,
        }

    model = NearestNeighbors(n_neighbors=2, metric="euclidean")
    model.fit(encoded_real)
    distances, _ = model.kneighbors(encoded_real)
    neighbor_distances = distances[:, 1]
    return {
        "distances": neighbor_distances,
        "mean_distance": float(np.mean(neighbor_distances)),
        "median_distance": float(np.median(neighbor_distances)),
        "p05_distance": float(np.quantile(neighbor_distances, 0.05)),
    }


def compute_nearest_source_distance(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    schema: Schema,
    max_rows: int = 2000,
) -> dict[str, float]:
    real_sample = real_df.sample(min(len(real_df), max_rows), random_state=42).reset_index(drop=True)
    syn_sample = syn_df.sample(min(len(syn_df), max_rows), random_state=42).reset_index(drop=True)

    combined = pd.concat([real_sample, syn_sample], axis=0, ignore_index=True)
    encoded = encode_for_distance(combined, schema)
    encoded_real = encoded.iloc[: len(real_sample)]
    encoded_syn = encoded.iloc[len(real_sample) :]

    if encoded_real.empty or encoded_syn.empty:
        return {"mean_distance": 0.0, "median_distance": 0.0, "p05_distance": 0.0}

    model = NearestNeighbors(n_neighbors=1, metric="euclidean")
    model.fit(encoded_real)
    distances, _ = model.kneighbors(encoded_syn)
    distances = distances[:, 0]
    return {
        "mean_distance": float(np.mean(distances)),
        "median_distance": float(np.median(distances)),
        "p05_distance": float(np.quantile(distances, 0.05)),
    }


def exact_match_rate(real_df: pd.DataFrame, syn_df: pd.DataFrame) -> float:
    real_records = set(_record_keys(real_df))
    syn_records = _record_keys(syn_df)
    return float(syn_records.isin(real_records).mean())


def duplicate_rate(syn_df: pd.DataFrame) -> float:
    if syn_df.empty:
        return 0.0
    duplicates = _record_keys(syn_df).duplicated().mean()
    return float(duplicates)


def privacy_score(real_df: pd.DataFrame, syn_df: pd.DataFrame, schema: Schema) -> dict[str, float]:
    distances = compute_nearest_source_distance(real_df, syn_df, schema)
    exact_rate = exact_match_rate(real_df, syn_df)
    dup_rate = duplicate_rate(syn_df)
    distance_component = distances["median_distance"] / (1.0 + distances["median_distance"])
    score = float(np.clip((1.0 - exact_rate) * (1.0 - dup_rate) * distance_component, 0.0, 1.0))
    return {
        "score": score,
        "exact_match_rate": exact_rate,
        "duplicate_rate": dup_rate,
        **distances,
    }


def filter_privacy_violations(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    schema: Schema,
    min_distance: float = 0.0,
    min_distance_quantile: float | None = None,
    max_rows: int = 2000,
) -> pd.DataFrame:
    filtered = syn_df.drop_duplicates().reset_index(drop=True)

    real_records = set(_record_keys(real_df))
    record_keys = _record_keys(filtered)
    filtered = filtered.loc[~record_keys.isin(real_records)].reset_index(drop=True)

    threshold_summary = resolve_privacy_min_distance(
        real_df,
        schema,
        min_distance=min_distance,
        min_distance_quantile=min_distance_quantile,
        max_rows=max_rows,
    )
    resolved_min_distance = float(threshold_summary["resolved_min_distance"])

    if resolved_min_distance <= 0.0 or filtered.empty:
        return filtered

    real_sample = real_df.sample(min(len(real_df), max_rows), random_state=42).reset_index(drop=True)
    combined = pd.concat([real_sample, filtered.reset_index(drop=True)], axis=0, ignore_index=True)
    encoded = encode_for_distance(combined, schema)
    encoded_real = encoded.iloc[: len(real_sample)]
    encoded_syn = encoded.iloc[len(real_sample) :]

    model = NearestNeighbors(n_neighbors=1, metric="euclidean")
    model.fit(encoded_real)
    distances, _ = model.kneighbors(encoded_syn)
    keep_mask = distances[:, 0] >= resolved_min_distance
    return filtered.loc[keep_mask].reset_index(drop=True)


def _record_keys(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(index=df.index, dtype="object")

    normalized = df.astype("object").where(df.notna(), "__MISSING__")
    rows = normalized.astype(str).to_numpy(dtype=str, copy=False)
    return pd.Series(["|".join(row) for row in rows], index=df.index, dtype="object")


def _scale_distance_features(encoded: pd.DataFrame) -> pd.DataFrame:
    if encoded.empty:
        return encoded

    scaled = encoded.astype(float).copy()
    for column in scaled.columns:
        series = pd.to_numeric(scaled[column], errors="coerce")
        non_null = series.dropna()
        if non_null.empty:
            scaled[column] = 0.0
            continue

        unique_values = set(np.unique(non_null.to_numpy(dtype=float)))
        if unique_values.issubset({0.0, 1.0}):
            scaled[column] = series.fillna(0.0)
            continue

        mean = float(non_null.mean())
        std = float(non_null.std(ddof=0))
        centered = series.fillna(mean) - mean
        if np.isfinite(std) and std > 1e-12:
            scaled[column] = centered / std
        else:
            scaled[column] = centered

    return scaled
