from __future__ import annotations

import pandas as pd

from src.utils.types import Schema


def find_candidate_derived_columns(df: pd.DataFrame, schema: Schema) -> list[str]:
    numeric_cols = schema.numeric_columns
    candidates: list[str] = []
    for column in numeric_cols:
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.notna().mean() > 0.8 and numeric.std(ddof=0) > 0:
            candidates.append(column)
    return candidates


def find_high_corr_pairs(df: pd.DataFrame, threshold: float = 0.95) -> list[tuple[str, str, float]]:
    numeric = df.apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr().fillna(0.0)
    pairs: list[tuple[str, str, float]] = []
    for i, left in enumerate(corr.columns):
        for right in corr.columns[i + 1 :]:
            value = float(corr.loc[left, right])
            if abs(value) >= threshold:
                pairs.append((left, right, value))
    return pairs


def find_exclusive_category_pairs(df: pd.DataFrame) -> list[tuple[str, str]]:
    columns = list(df.columns)
    pairs: list[tuple[str, str]] = []
    for i, left in enumerate(columns):
        left_values = set(df[left].dropna().astype("object"))
        if len(left_values) > 12 or not left_values:
            continue
        for right in columns[i + 1 :]:
            right_values = set(df[right].dropna().astype("object"))
            if len(right_values) > 12 or not right_values:
                continue
            if left_values.isdisjoint(right_values):
                pairs.append((left, right))
    return pairs


def suggest_constraints(df: pd.DataFrame, schema: Schema) -> dict:
    constraints = {
        "ranges": {},
        "allowed_values": {},
        "derived": [],
        "impossible_combinations": [],
    }

    for column in schema.columns:
        if column.kind == "numeric":
            numeric = pd.to_numeric(df[column.name], errors="coerce").dropna()
            if not numeric.empty:
                constraints["ranges"][column.name] = [float(numeric.min()), float(numeric.max())]
        elif column.kind in {"categorical", "binary"} and column.unique_count <= 50:
            constraints["allowed_values"][column.name] = list(
                df[column.name].dropna().astype("object").drop_duplicates().tolist()
            )

    return constraints

