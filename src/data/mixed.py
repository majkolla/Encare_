from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

VALUE_STATE_TOKEN = "__VALUE__"
MISSING_STATE_TOKEN = "__MISSING__"


@dataclass
class MixedColumnEncoding:
    name: str
    source_kind: str
    value_column: str
    state_column: str
    numeric_support: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    numeric_support_strings: list[str] = field(default_factory=list)


def split_mixed_column(
    column_name: str,
    series: pd.Series,
    source_kind: str,
    snap_numeric_max_unique: int = 64,
) -> tuple[MixedColumnEncoding, pd.Series, pd.Series]:
    value_series = coerce_mixed_value_series(series, source_kind)
    token_mask = series.notna() & value_series.isna()
    state_series = pd.Series(VALUE_STATE_TOKEN, index=series.index, dtype="object")
    state_series.loc[series.isna()] = MISSING_STATE_TOKEN
    state_series.loc[token_mask] = (
        series.loc[token_mask]
        .astype("object")
        .astype(str)
        .str.strip()
    )

    numeric_support = np.array([], dtype=float)
    numeric_support_strings: list[str] = []
    if source_kind == "numeric":
        numeric_support, numeric_support_strings = observed_numeric_string_support(
            original_series=series,
            numeric_series=value_series,
            snap_numeric_max_unique=snap_numeric_max_unique,
        )

    encoding = MixedColumnEncoding(
        name=column_name,
        source_kind=source_kind,
        value_column=f"{column_name}__value",
        state_column=f"{column_name}__state",
        numeric_support=numeric_support,
        numeric_support_strings=numeric_support_strings,
    )
    return encoding, value_series, state_series


def coerce_mixed_value_series(series: pd.Series, source_kind: str) -> pd.Series:
    normalized_kind = str(source_kind).lower()
    if normalized_kind == "datetime":
        return datetime_to_numeric(series)
    if normalized_kind == "time":
        return time_to_numeric(series)
    return pd.to_numeric(series, errors="coerce")


def restore_mixed_column(
    sampled_df: pd.DataFrame,
    encoding: MixedColumnEncoding,
) -> pd.Series:
    if encoding.state_column not in sampled_df.columns:
        return pd.Series(np.nan, index=sampled_df.index, dtype="object")

    state_series = sampled_df[encoding.state_column].astype("object")
    value_series = sampled_df.get(
        encoding.value_column,
        pd.Series(np.nan, index=sampled_df.index, dtype="float64"),
    )

    if encoding.source_kind == "numeric":
        restored_values = restore_numeric_string_support(
            value_series,
            encoding.numeric_support,
            encoding.numeric_support_strings,
        )
    elif encoding.source_kind == "datetime":
        restored_values = numeric_to_datetime(value_series).astype("object")
    elif encoding.source_kind == "time":
        restored_values = value_series.apply(seconds_to_time).astype("object")
    else:
        restored_values = value_series.astype("object")

    restored = restored_values.where(state_series.eq(VALUE_STATE_TOKEN), state_series)
    restored = restored.where(~state_series.eq(MISSING_STATE_TOKEN), np.nan)
    return pd.Series(restored, index=sampled_df.index, dtype="object")


def numeric_support_values(
    series: pd.Series | np.ndarray,
    unique_count: int,
    snap_numeric_max_unique: int,
) -> np.ndarray:
    if unique_count > snap_numeric_max_unique:
        return np.array([], dtype=float)

    observed = pd.to_numeric(pd.Series(series), errors="coerce").dropna().unique()
    if len(observed) == 0 or len(observed) > snap_numeric_max_unique:
        return np.array([], dtype=float)

    return np.sort(np.asarray(observed, dtype=float))


def observed_numeric_string_support(
    original_series: pd.Series,
    numeric_series: pd.Series,
    snap_numeric_max_unique: int = 64,
) -> tuple[np.ndarray, list[str]]:
    numeric = pd.to_numeric(numeric_series, errors="coerce")
    valid_mask = original_series.notna() & numeric.notna()
    if not valid_mask.any():
        return np.array([], dtype=float), []

    observed = pd.DataFrame(
        {
            "numeric_value": numeric.loc[valid_mask].astype("float64"),
            "original_value": (
                original_series.loc[valid_mask]
                .astype("object")
                .astype(str)
                .str.strip()
            ),
        }
    )
    if observed.empty or observed["numeric_value"].nunique() > snap_numeric_max_unique:
        return np.array([], dtype=float), []

    representatives = (
        observed.groupby(["numeric_value", "original_value"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(
            by=["numeric_value", "count", "original_value"],
            ascending=[True, False, True],
        )
        .drop_duplicates(subset=["numeric_value"], keep="first")
        .sort_values(by="numeric_value")
        .reset_index(drop=True)
    )
    return (
        representatives["numeric_value"].to_numpy(dtype=float),
        representatives["original_value"].tolist(),
    )


def restore_numeric_string_support(
    series: pd.Series,
    support_values: np.ndarray,
    support_strings: list[str],
) -> pd.Series:
    snapped = snap_numeric_to_support(series, support_values)
    if len(support_values) == 0 or len(support_values) != len(support_strings):
        return snapped.astype("object")

    lookup = {
        float(numeric_value): string_value
        for numeric_value, string_value in zip(support_values, support_strings)
    }
    restored_values: list[object] = []
    for value in snapped.to_numpy(dtype=float, na_value=np.nan):
        if np.isnan(value):
            restored_values.append(np.nan)
            continue
        restored_values.append(lookup.get(float(value), value))

    return pd.Series(restored_values, index=series.index, dtype="object")


def snap_numeric_to_support(
    series: pd.Series | np.ndarray,
    support_values: np.ndarray | None,
) -> pd.Series:
    numeric = pd.to_numeric(pd.Series(series), errors="coerce")
    if support_values is None or len(support_values) == 0 or numeric.dropna().empty:
        return numeric

    snapped = numeric.to_numpy(dtype=float, na_value=np.nan)
    valid_mask = ~np.isnan(snapped)
    valid_values = snapped[valid_mask]
    insert_positions = np.searchsorted(support_values, valid_values, side="left")
    insert_positions = np.clip(insert_positions, 0, len(support_values) - 1)
    left_positions = np.clip(insert_positions - 1, 0, len(support_values) - 1)
    right_positions = insert_positions
    left_values = support_values[left_positions]
    right_values = support_values[right_positions]
    choose_left = np.abs(valid_values - left_values) <= np.abs(valid_values - right_values)
    snapped_values = np.where(choose_left, left_values, right_values)
    snapped[valid_mask] = snapped_values
    return pd.Series(snapped, index=numeric.index, dtype="float64")


def datetime_to_numeric(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    numeric = pd.Series(parsed.astype("int64"), index=series.index, dtype="float64")
    numeric[parsed.isna()] = np.nan
    return numeric / 1_000_000_000


def numeric_to_datetime(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    parsed = pd.to_datetime(numeric, unit="s", errors="coerce")
    formatted = parsed.dt.strftime("%Y-%m-%d")
    return formatted.where(parsed.notna(), np.nan)


def time_to_numeric(series: pd.Series) -> pd.Series:
    text = series.astype(str).where(series.notna(), None)
    split = text.str.split(":", expand=True)
    hours = pd.to_numeric(split[0], errors="coerce")
    minutes = pd.to_numeric(split[1], errors="coerce")
    return hours * 3600 + minutes * 60


def seconds_to_time(value: float | int | None) -> str | float:
    if value is None or pd.isna(value):
        return np.nan

    total_seconds = int(round(float(value)))
    hours = max(min(total_seconds // 3600, 23), 0)
    minutes = max(min((total_seconds % 3600) // 60, 59), 0)
    return f"{hours:02d}:{minutes:02d}"
