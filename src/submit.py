from __future__ import annotations

from pathlib import Path

import numpy as np
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
                column,
            )

    return submission[schema.column_order]


def _coerce_object_column(
    submission_series: pd.Series,
    reference_series: pd.Series,
    column_schema,
) -> pd.Series:
    series = submission_series.astype("object").copy()
    reference_non_null = reference_series.dropna().astype("object")
    if reference_non_null.empty:
        return series

    if getattr(column_schema, "mixed_value_kind", None) is None:
        if series.isna().all():
            anchor_value = reference_non_null.mode(dropna=True)
            if not anchor_value.empty and len(series) > 0:
                series.iloc[0] = anchor_value.iloc[0]
        return series

    if series.isna().all():
        target_non_null = _scaled_target_count(
            observed_count=len(reference_non_null),
            observed_total=len(reference_series),
            target_total=len(series),
        )
        fill_indices = _spread_indices(series.index.tolist(), target_non_null)
        fill_values = _expanded_reference_values(
            reference_non_null.value_counts(dropna=True, sort=False),
            len(fill_indices),
        )
        if fill_indices and fill_values:
            series.loc[fill_indices] = fill_values
        return series

    reference_non_numeric = _reference_non_numeric_counts(reference_non_null)
    if reference_non_numeric.empty:
        return series

    desired_non_numeric = _scaled_target_count(
        observed_count=int(reference_non_numeric.sum()),
        observed_total=len(reference_series),
        target_total=len(series),
    )
    current_non_numeric = _current_non_numeric_count(series)
    deficit = max(desired_non_numeric - current_non_numeric, 0)
    if deficit == 0:
        return series

    candidate_indices = series[series.isna()].index.tolist()
    if len(candidate_indices) < deficit:
        existing_candidates = set(candidate_indices)
        candidate_indices.extend(
            idx for idx in series.index.tolist()
            if idx not in existing_candidates and _looks_numeric_like(series.loc[idx])
        )
    if len(candidate_indices) < deficit:
        existing_candidates = set(candidate_indices)
        candidate_indices.extend(idx for idx in series.index.tolist() if idx not in existing_candidates)

    fill_indices = _spread_indices(candidate_indices, deficit)
    fill_values = _expanded_reference_values(reference_non_numeric, len(fill_indices))
    if fill_indices and fill_values:
        series.loc[fill_indices] = fill_values
    return series


def _scaled_target_count(
    observed_count: int,
    observed_total: int,
    target_total: int,
) -> int:
    if observed_count <= 0 or observed_total <= 0 or target_total <= 0:
        return 0

    scaled = int(round((observed_count / observed_total) * target_total))
    return max(1, min(target_total, scaled))


def _reference_non_numeric_counts(reference_non_null: pd.Series) -> pd.Series:
    values = reference_non_null.astype(str)
    parsed = pd.to_numeric(values, errors="coerce")
    non_numeric_mask = parsed.isna()
    if not non_numeric_mask.any():
        return pd.Series(dtype="int64")
    return values[non_numeric_mask].value_counts(dropna=True, sort=False)


def _current_non_numeric_count(series: pd.Series) -> int:
    non_null = series.dropna()
    if non_null.empty:
        return 0

    as_text = non_null.astype(str)
    parsed = pd.to_numeric(as_text, errors="coerce")
    return int(parsed.isna().sum())


def _looks_numeric_like(value: object) -> bool:
    if pd.isna(value):
        return False
    return pd.notna(pd.to_numeric(value, errors="coerce"))


def _spread_indices(indices: list[int], count: int) -> list[int]:
    if count <= 0 or not indices:
        return []
    if count >= len(indices):
        return indices

    positions = np.linspace(0, len(indices) - 1, num=count, dtype=int)
    return [indices[pos] for pos in positions]


def _expanded_reference_values(value_counts: pd.Series, count: int) -> list[object]:
    if count <= 0 or value_counts.empty:
        return []

    probabilities = value_counts.astype(float) / float(value_counts.sum())
    raw_allocations = probabilities * count
    base_allocations = np.floor(raw_allocations).astype(int)
    remainder = count - int(base_allocations.sum())

    if remainder > 0:
        remainders = raw_allocations - base_allocations
        order = np.argsort(-remainders.to_numpy(dtype=float), kind="stable")
        for idx in order[:remainder]:
            base_allocations.iloc[int(idx)] += 1

    expanded: list[object] = []
    for value, allocation in zip(value_counts.index.tolist(), base_allocations.tolist()):
        expanded.extend([value] * int(allocation))

    return expanded[:count]


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
