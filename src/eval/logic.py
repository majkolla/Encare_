from __future__ import annotations

import numpy as np
import pandas as pd


def conditional_blank_violations(df: pd.DataFrame, constraints: dict) -> dict[str, float]:
    violations: dict[str, float] = {}
    for rule in constraints.get("conditional_blanks", []):
        parent = rule.get("parent")
        if not parent or parent not in df.columns:
            continue

        inactive_mask = _conditional_inactive_mask(df[parent], rule)
        if not inactive_mask.any():
            violations[str(parent)] = 0.0
            continue

        child_rates: list[float] = []
        for child in rule.get("children", []):
            if child not in df.columns:
                continue
            child_rates.append(float(df.loc[inactive_mask, child].notna().mean()))

        violations[str(parent)] = float(np.mean(child_rates)) if child_rates else 0.0
    return violations


def range_violations(df: pd.DataFrame, constraints: dict) -> dict[str, float]:
    violations: dict[str, float] = {}
    for column, (lower, upper) in constraints.get("ranges", {}).items():
        if column not in df.columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        valid = numeric.between(lower, upper) | numeric.isna()
        violations[column] = float((~valid).mean())
    return violations


def category_violations(df: pd.DataFrame, constraints: dict) -> dict[str, float]:
    violations: dict[str, float] = {}
    for column, allowed_values in constraints.get("allowed_values", {}).items():
        if column not in df.columns or not allowed_values:
            continue
        valid = df[column].isin(allowed_values) | df[column].isna()
        violations[column] = float((~valid).mean())
    return violations


def derived_field_violations(df: pd.DataFrame, constraints: dict) -> dict[str, float]:
    violations: dict[str, float] = {}
    for rule in constraints.get("derived", []):
        target = rule["target"]
        inputs = rule["inputs"]
        if target not in df.columns or any(column not in df.columns for column in inputs):
            continue
        violations[target] = float(derived_field_violation_mask(df, rule).mean())
    return violations


def derived_field_violation_mask(df: pd.DataFrame, rule: dict) -> pd.Series:
    target = rule["target"]
    expected = _coerce_expected_series(rule["fn"](df), df.index)
    tolerance = float(rule.get("tolerance", 0.0))
    kind = str(rule.get("kind", "numeric")).lower()

    if kind == "datetime":
        actual = pd.to_datetime(df[target], errors="coerce")
        expected_values = pd.to_datetime(expected, errors="coerce")
        difference = (actual - expected_values).abs() / pd.Timedelta(days=1)
        valid = actual.isna() | expected_values.isna() | (difference <= tolerance)
        return ~valid

    if kind == "time":
        actual = _time_to_seconds(df[target])
        expected_values = _time_to_seconds(expected)
        valid = actual.isna() | expected_values.isna() | ((actual - expected_values).abs() <= tolerance)
        return ~valid

    if kind == "categorical":
        actual = df[target].astype("object")
        expected_values = expected.astype("object")
        valid = actual.isna() | expected_values.isna() | actual.eq(expected_values)
        return ~valid

    actual = pd.to_numeric(df[target], errors="coerce")
    expected_values = pd.to_numeric(expected, errors="coerce")
    valid = actual.isna() | expected_values.isna() | ((actual - expected_values).abs() <= tolerance)
    return ~valid


def logic_score(df: pd.DataFrame, constraints: dict) -> dict[str, float | dict[str, float]]:
    ranges = range_violations(df, constraints)
    categories = category_violations(df, constraints)
    derived = derived_field_violations(df, constraints)
    conditional_blanks = conditional_blank_violations(df, constraints)

    rates = list(ranges.values()) + list(categories.values()) + list(derived.values()) + list(conditional_blanks.values())
    violation_rate = float(np.mean(rates)) if rates else 0.0
    return {
        "score": max(0.0, 1.0 - violation_rate),
        "violation_rate": violation_rate,
        "details": {
            "ranges": ranges,
            "categories": categories,
            "derived": derived,
            "conditional_blanks": conditional_blanks,
        },
    }


def _coerce_expected_series(
    values: pd.Series | np.ndarray | list | object,
    index: pd.Index | None,
) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.reindex(index) if index is not None else values
    return pd.Series(values, index=index)


def _time_to_seconds(values: pd.Series | np.ndarray | list | object) -> pd.Series:
    index = values.index if isinstance(values, pd.Series) else None
    series = _coerce_expected_series(values, index)
    text = series.astype("object").where(series.notna(), None).astype(str)
    split = text.str.split(":", expand=True)
    hours = pd.to_numeric(split[0], errors="coerce")
    minutes = pd.to_numeric(split[1], errors="coerce")
    return hours * 3600 + minutes * 60


def _conditional_inactive_mask(parent_series: pd.Series, rule: dict) -> pd.Series:
    values = parent_series.astype("object")
    inactive_mask = pd.Series(False, index=parent_series.index)

    inactive_values = {str(value) for value in rule.get("inactive_values", [])}
    if inactive_values:
        inactive_mask |= values.astype(str).isin(inactive_values)

    prefixes = [str(prefix) for prefix in rule.get("inactive_prefixes", []) if str(prefix)]
    if prefixes:
        parent_text = values.astype(str)
        for prefix in prefixes:
            inactive_mask |= parent_text.str.startswith(prefix)

    return inactive_mask
