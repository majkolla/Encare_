from __future__ import annotations

import re
from typing import Any

import pandas as pd

from src.utils.types import ColumnSchema, Schema

_TIME_PATTERN = re.compile(r"^\d{2}:\d{2}$")
_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATE_HINT_PATTERN = re.compile(r"(yyyy-mm-dd|date)", re.IGNORECASE)
_TIME_HINT_PATTERN = re.compile(r"hh:mm", re.IGNORECASE)
_ID_HINT_PATTERN = re.compile(r"(^id$|_id$|patient id|identifier)", re.IGNORECASE)
_MIXED_NUMERIC_HINT_PATTERN = re.compile(
    r"(age|bmi|weight|height|duration|time|length|stay|score|dose|volume|fluid|blood|days?|nights?|kg|cm|ml)",
    re.IGNORECASE,
)


def detect_id_like_columns(df: pd.DataFrame) -> list[str]:
    id_like = []
    n_rows = max(len(df), 1)

    for column in df.columns:
        series = df[column]
        non_null = series.dropna()
        if non_null.empty:
            continue

        unique_count = non_null.nunique(dropna=True)
        unique_ratio = unique_count / n_rows
        monotonic = False

        numeric_version = pd.to_numeric(non_null, errors="coerce")
        if numeric_version.notna().all():
            monotonic = numeric_version.is_monotonic_increasing

        if unique_ratio >= 0.98 or monotonic or _ID_HINT_PATTERN.search(column):
            id_like.append(column)

    return id_like


def detect_low_cardinality_columns(df: pd.DataFrame, max_unique: int = 12) -> list[str]:
    low_cardinality = []
    for column in df.columns:
        unique_count = df[column].dropna().nunique(dropna=True)
        if 0 < unique_count <= max_unique:
            low_cardinality.append(column)
    return low_cardinality


def build_missingness_mask(df: pd.DataFrame) -> pd.DataFrame:
    return df.isna().copy()


def infer_schema(df: pd.DataFrame) -> Schema:
    id_like_columns = set(detect_id_like_columns(df))
    low_cardinality_columns = set(detect_low_cardinality_columns(df))

    columns: list[ColumnSchema] = []
    row_count = len(df)

    for column in df.columns:
        series = df[column]
        non_null = series.dropna()
        unique_count = non_null.nunique(dropna=True)
        missing_rate = float(series.isna().mean())
        pandas_dtype = str(series.dtype)
        nullable = bool(series.isna().any())

        numeric_version = pd.to_numeric(non_null, errors="coerce")
        numeric_like_ratio = (
            float(numeric_version.notna().mean()) if len(non_null) > 0 else 0.0
        )
        explicit_non_numeric_tokens = (
            bool(len(non_null) > 0)
            and not pd.api.types.is_numeric_dtype(series)
            and bool(numeric_version.notna().any())
            and bool(numeric_version.isna().any())
        )
        numeric_like = numeric_like_ratio >= 0.80 or (
            numeric_like_ratio >= 0.50 and unique_count >= 10
        )
        mixed_value_kind = _infer_mixed_value_kind(
            column_name=column,
            numeric_like_ratio=numeric_like_ratio,
            unique_count=unique_count,
            explicit_non_numeric_tokens=explicit_non_numeric_tokens,
        )

        if unique_count <= 1:
            kind = "constant"
        elif _looks_like_datetime(column, non_null):
            kind = "datetime"
        elif _looks_like_time(column, non_null):
            kind = "time"
        elif column in id_like_columns:
            kind = "id_like"
        elif explicit_non_numeric_tokens:
            kind = "binary" if unique_count <= 2 else "categorical"
        elif pd.api.types.is_numeric_dtype(series) or numeric_like:
            kind = "binary" if unique_count <= 2 else "numeric"
        else:
            kind = "binary" if unique_count <= 2 else "categorical"

        min_value = None
        max_value = None
        if kind == "numeric" and not numeric_version.dropna().empty:
            min_value = float(numeric_version.min())
            max_value = float(numeric_version.max())

        allowed_values: list[Any] = []
        if kind in {"categorical", "binary", "id_like"} and unique_count <= 100:
            allowed_values = list(non_null.astype("object").drop_duplicates().tolist())

        mixed_token_values: list[Any] = []
        if mixed_value_kind is not None:
            non_numeric_tokens = non_null[pd.to_numeric(non_null, errors="coerce").isna()]
            mixed_token_values = list(
                non_numeric_tokens.astype("object").drop_duplicates().head(100).tolist()
            )

        columns.append(
            ColumnSchema(
                name=column,
                kind=kind,
                pandas_dtype=pandas_dtype,
                nullable=nullable,
                unique_count=int(unique_count),
                missing_rate=missing_rate,
                numeric_like=numeric_like,
                low_cardinality=column in low_cardinality_columns,
                id_like=column in id_like_columns,
                min_value=min_value,
                max_value=max_value,
                allowed_values=allowed_values,
                mixed_value_kind=mixed_value_kind,
                mixed_token_values=mixed_token_values,
            )
        )

    return Schema(columns=columns, column_order=list(df.columns), row_count=row_count)


def _looks_like_datetime(column_name: str, non_null: pd.Series) -> bool:
    if _DATE_HINT_PATTERN.search(column_name):
        return True
    if non_null.empty:
        return False
    sample = non_null.astype(str).head(200)
    matches = sample.str.match(_DATE_PATTERN)
    return float(matches.mean()) >= 0.95


def _looks_like_time(column_name: str, non_null: pd.Series) -> bool:
    if _TIME_HINT_PATTERN.search(column_name):
        sample = non_null.astype(str).head(200)
        matches = sample.str.match(_TIME_PATTERN)
        return float(matches.mean()) >= 0.95
    if non_null.empty:
        return False
    matches = non_null.astype(str).head(200).str.match(_TIME_PATTERN)
    return float(matches.mean()) >= 0.95


def _infer_mixed_value_kind(
    column_name: str,
    numeric_like_ratio: float,
    unique_count: int,
    explicit_non_numeric_tokens: bool,
) -> str | None:
    if not explicit_non_numeric_tokens:
        return None

    if numeric_like_ratio >= 0.50:
        return "numeric"

    if numeric_like_ratio >= 0.20 and unique_count >= 10 and _MIXED_NUMERIC_HINT_PATTERN.search(column_name):
        return "numeric"

    return None
