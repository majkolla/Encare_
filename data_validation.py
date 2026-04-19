import pandas as pd


def validate_submission(original_df: pd.DataFrame, submission_df: pd.DataFrame) -> list[str]:
    errors = []

    # 1. Column count
    if len(submission_df.columns) != len(original_df.columns):
        original_col_set = set(original_df.columns)
        submission_col_set = set(submission_df.columns)
        missing = sorted(original_col_set - submission_col_set)
        extra = sorted(submission_col_set - original_col_set)
        detail_parts = []
        if missing:
            detail_parts.append(f"missing: {', '.join(missing)}")
        if extra:
            detail_parts.append(f"extra: {', '.join(extra)}")
        detail = f" ({'; '.join(detail_parts)})" if detail_parts else ""
        errors.append(
            f"Column count mismatch: expected {len(original_df.columns)}, got {len(submission_df.columns)}{detail}"
        )
        # Cannot meaningfully validate names/types if count differs
        return errors

    # 2. Column names
    original_cols = list(original_df.columns)
    submission_cols = list(submission_df.columns)
    if submission_cols != original_cols:
        mismatched = [
            f"position {i}: '{s}' (expected '{o}')"
            for i, (o, s) in enumerate(zip(original_cols, submission_cols))
            if o != s
        ]
        errors.append(f"Column name mismatch: {'; '.join(mismatched[:5])}")

    # 3. Data types — compare dtype.kind (i=int, f=float, O=object, b=bool, M=datetime)
    for col in original_df.columns:
        if col not in submission_df.columns:
            continue
        orig_kind = original_df[col].dtype.kind
        sub_kind = submission_df[col].dtype.kind
        # Allow int/float interchangeably (pandas may infer float when NaNs present)
        numeric_kinds = {"i", "u", "f"}
        if orig_kind in numeric_kinds and sub_kind in numeric_kinds:
            continue
        if orig_kind != sub_kind:
            errors.append(
                f"Column '{col}' type mismatch: expected ({original_df[col].dtype}), got ({submission_df[col].dtype})"
            )

    # 4. Minimum row count (must have at least as many rows as the original)
    min_rows = len(original_df)
    if len(submission_df) < min_rows:
        errors.append(
            f"Too few rows: expected at least {min_rows}, got {len(submission_df)}"
        )

    return errors
