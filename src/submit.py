from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loader import save_submission_csv
from src.utils.io import ensure_dir
from src.utils.types import Schema


def ensure_exact_schema(df: pd.DataFrame, reference_df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    submission = df.copy()
    submission = submission.reindex(columns=schema.column_order)

    for column in schema.columns:
        if column.name not in submission.columns:
            submission[column.name] = pd.NA
            continue

        reference_kind = reference_df[column.name].dtype.kind

        if reference_kind in {"i", "u", "f"}:
            submission[column.name] = pd.to_numeric(submission[column.name], errors="coerce")
        else:
            submission[column.name] = _coerce_object_column(
                submission[column.name],
                reference_df[column.name],
            )

    return submission[schema.column_order]


def _coerce_object_column(submission_series: pd.Series, reference_series: pd.Series) -> pd.Series:
    series = submission_series.astype("object").copy()
    reference_non_null = reference_series.dropna()

    if series.isna().all() and not reference_non_null.empty:
        mode = reference_non_null.mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else reference_non_null.iloc[0]
        return pd.Series([fill_value] * len(series), index=series.index, dtype="object")

    non_numeric_token = _find_non_numeric_token(reference_non_null)
    if non_numeric_token is None:
        return series

    non_null_values = series.dropna().astype(str)
    has_non_numeric = False
    if not non_null_values.empty:
        parsed = pd.to_numeric(non_null_values, errors="coerce")
        has_non_numeric = bool(parsed.isna().any())

    if has_non_numeric:
        return series

    if series.isna().any():
        replacement_index = series[series.isna()].index[0]
    else:
        replacement_index = series.index[0]
    series.loc[replacement_index] = non_numeric_token
    return series


def _find_non_numeric_token(reference_non_null: pd.Series) -> str | None:
    if reference_non_null.empty:
        return None

    values = reference_non_null.astype(str)
    parsed = pd.to_numeric(values, errors="coerce")
    non_numeric = values[parsed.isna()]
    if non_numeric.empty:
        return None

    mode = non_numeric.mode(dropna=True)
    if not mode.empty:
        return str(mode.iloc[0])
    return str(non_numeric.iloc[0])


def validate_submission(reference_df: pd.DataFrame, submission_df: pd.DataFrame) -> list[str]:
    errors: list[str] = []

    if list(submission_df.columns) != list(reference_df.columns):
        errors.append("Column order or column names do not match the source dataset.")

    if len(submission_df) < len(reference_df):
        errors.append(f"Too few rows: expected at least {len(reference_df)}, got {len(submission_df)}.")

    numeric_kinds = {"i", "u", "f"}
    for column in reference_df.columns:
        if column not in submission_df.columns:
            continue
        reference_kind = reference_df[column].dtype.kind
        submission_kind = submission_df[column].dtype.kind
        if reference_kind in numeric_kinds and submission_kind in numeric_kinds:
            continue
        if reference_kind != submission_kind:
            errors.append(
                f"Column '{column}' type mismatch: expected {reference_df[column].dtype}, got {submission_df[column].dtype}."
            )

    return errors


def save_validated_submission(
    df: pd.DataFrame,
    reference_df: pd.DataFrame,
    schema: Schema,
    output_path: str | Path,
) -> tuple[Path, list[str]]:
    ensure_dir(Path(output_path).parent)
    submission = ensure_exact_schema(df, reference_df, schema)
    save_submission_csv(submission, output_path)
    round_tripped = pd.read_csv(output_path, low_memory=False)
    errors = validate_submission(reference_df, round_tripped)
    return Path(output_path), errors
