from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Sequence

from src.eval.logic import (
    category_violations,
    derived_field_violation_mask,
    derived_field_violations,
    range_violations,
)


def clip_ranges(df: pd.DataFrame, constraints: dict) -> pd.DataFrame:
    repaired = df.copy()
    for column, (lower, upper) in constraints.get("ranges", {}).items():
        if column not in repaired.columns:
            continue
        numeric = pd.to_numeric(repaired[column], errors="coerce")
        repaired[column] = numeric.clip(lower, upper)
    return repaired


def normalize_categories(df: pd.DataFrame, constraints: dict) -> pd.DataFrame:
    repaired = df.copy()
    for column, allowed_values in constraints.get("allowed_values", {}).items():
        if column not in repaired.columns or not allowed_values:
            continue
        repaired.loc[~repaired[column].isin(allowed_values), column] = np.nan
    return repaired


def enforce_conditional_blanks(df: pd.DataFrame, constraints: dict) -> pd.DataFrame:
    repaired = df.copy()
    for rule in constraints.get("conditional_blanks", []):
        parent = rule["parent"]
        if parent not in repaired.columns:
            continue

        inactive_mask = _conditional_inactive_mask(repaired[parent], rule)
        if not inactive_mask.any():
            continue

        for child in rule.get("children", []):
            if child not in repaired.columns:
                continue
            repaired.loc[inactive_mask, child] = np.nan

    return repaired


def softly_align_conditional_blank_rates(
    syn_df: pd.DataFrame,
    real_df: pd.DataFrame,
    constraints: dict,
    strength: float = 0.0,
    min_excess_rate: float = 0.05,
    include_parents: Sequence[str] | None = None,
    exclude_parents: Sequence[str] | None = None,
    include_children: Sequence[str] | None = None,
    exclude_children: Sequence[str] | None = None,
    max_blanks_per_rule: int | None = None,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    clipped_strength = float(np.clip(strength, 0.0, 1.0))
    min_excess = max(float(min_excess_rate), 0.0)
    if clipped_strength <= 0.0:
        return syn_df.copy(), []

    adjusted = syn_df.copy()
    summary_rows: list[dict[str, object]] = []
    include_terms = [str(value).strip().lower() for value in (include_parents or []) if str(value).strip()]
    exclude_terms = [str(value).strip().lower() for value in (exclude_parents or []) if str(value).strip()]
    include_child_terms = [str(value).strip().lower() for value in (include_children or []) if str(value).strip()]
    exclude_child_terms = [str(value).strip().lower() for value in (exclude_children or []) if str(value).strip()]

    for rule in constraints.get("conditional_blanks", []):
        parent = rule.get("parent")
        if not parent or parent not in adjusted.columns or parent not in real_df.columns:
            continue
        if not _rule_parent_selected(str(parent), include_terms, exclude_terms):
            continue

        real_inactive_mask = _conditional_inactive_mask(real_df[parent], rule)
        syn_inactive_mask = _conditional_inactive_mask(adjusted[parent], rule)
        syn_inactive_count = int(syn_inactive_mask.sum())
        if syn_inactive_count == 0:
            continue

        for child in rule.get("children", []):
            if child not in adjusted.columns or child not in real_df.columns:
                continue
            if not _rule_parent_selected(str(child), include_child_terms, exclude_child_terms):
                continue

            real_target_rate = (
                float(real_df.loc[real_inactive_mask, child].notna().mean())
                if real_inactive_mask.any()
                else 0.0
            )
            syn_present_indices = adjusted.index[syn_inactive_mask & adjusted[child].notna()].tolist()
            syn_present_count = len(syn_present_indices)
            if syn_present_count == 0:
                continue

            syn_rate = float(syn_present_count / syn_inactive_count)
            excess_rate = syn_rate - real_target_rate
            if excess_rate <= min_excess:
                continue

            target_rate = max(real_target_rate, syn_rate - clipped_strength * excess_rate)
            target_present_count = int(round(target_rate * syn_inactive_count))
            blank_count = syn_present_count - target_present_count
            if max_blanks_per_rule is not None:
                blank_count = min(blank_count, max(int(max_blanks_per_rule), 0))
            if blank_count <= 0:
                continue

            selected_indices = _spread_selection(syn_present_indices, blank_count)
            adjusted.loc[selected_indices, child] = np.nan
            summary_rows.append(
                {
                    "parent": str(parent),
                    "child": str(child),
                    "syn_rate_before": syn_rate,
                    "real_rate": real_target_rate,
                    "target_rate": target_rate,
                    "blanked_rows": len(selected_indices),
                }
            )

    return adjusted, summary_rows


def recompute_derived_fields(df: pd.DataFrame, constraints: dict) -> pd.DataFrame:
    repaired = df.copy()
    repair_options = constraints.get("repair_options", {})
    derived_repair_mode = str(repair_options.get("derived_repair_mode", "overwrite")).lower()
    for rule in constraints.get("derived", []):
        target = rule["target"]
        if target not in repaired.columns:
            continue
        expected = rule["fn"](repaired)
        kind = str(rule.get("kind", "numeric")).lower()

        if derived_repair_mode == "fill_missing":
            repaired[target] = _fill_missing_target(repaired[target], expected)
            continue

        if derived_repair_mode == "fill_missing_datetime" and kind in {"datetime", "time"}:
            repaired[target] = _fill_missing_target(repaired[target], expected)
            continue

        repaired[target] = expected
    repaired = align_derived_missingness(repaired, constraints)
    return repaired


def _fill_missing_target(actual: pd.Series, expected: pd.Series | np.ndarray | list | object) -> pd.Series:
    expected_series = expected if isinstance(expected, pd.Series) else pd.Series(expected, index=actual.index)
    return actual.where(actual.notna(), expected_series)


def align_derived_missingness(df: pd.DataFrame, constraints: dict) -> pd.DataFrame:
    repaired = df.copy()
    repair_options = constraints.get("repair_options", {})
    derived_repair_mode = str(repair_options.get("derived_repair_mode", "overwrite")).lower()
    if derived_repair_mode not in {"fill_missing", "fill_missing_datetime"}:
        return repaired

    for rule in constraints.get("derived", []):
        target = rule.get("target")
        kind = str(rule.get("kind", "numeric")).lower()
        target_missing_rate = rule.get("target_missing_rate")
        if (
            target not in repaired.columns
            or target_missing_rate is None
            or kind not in {"datetime", "time"}
        ):
            continue

        desired_missing = int(round(float(target_missing_rate) * len(repaired)))
        current_missing_mask = repaired[target].isna()
        deficit = desired_missing - int(current_missing_mask.sum())
        if deficit <= 0:
            continue

        preferred_candidates = _missing_input_candidates(repaired, rule, current_missing_mask)
        selected = _spread_selection(preferred_candidates, deficit)
        if len(selected) < deficit:
            fallback_candidates = repaired.index[~current_missing_mask].difference(pd.Index(selected))
            selected.extend(_spread_selection(fallback_candidates.tolist(), deficit - len(selected)))

        if selected:
            repaired.loc[selected, target] = np.nan

    return repaired


def _missing_input_candidates(df: pd.DataFrame, rule: dict, current_missing_mask: pd.Series) -> list[int]:
    inputs = [column for column in rule.get("inputs", []) if column in df.columns]
    if len(inputs) <= 1:
        return df.index[~current_missing_mask].tolist()

    auxiliary_inputs = inputs[1:]
    aux_missing_mask = pd.Series(False, index=df.index)
    for column in auxiliary_inputs:
        aux_missing_mask |= df[column].isna()

    preferred = df.index[aux_missing_mask & ~current_missing_mask]
    return preferred.tolist()


def _spread_selection(indices: list[int], count: int) -> list[int]:
    if count <= 0 or not indices:
        return []
    if count >= len(indices):
        return list(indices)

    positions = np.linspace(0, len(indices) - 1, num=count, dtype=int)
    return [indices[pos] for pos in positions]


def _conditional_inactive_mask(parent_series: pd.Series, rule: dict) -> pd.Series:
    parent_values = parent_series.astype("object")
    inactive_mask = pd.Series(False, index=parent_series.index)

    inactive_values = {str(value) for value in rule.get("inactive_values", [])}
    if inactive_values:
        inactive_mask |= parent_values.astype(str).isin(inactive_values)

    prefixes = [str(prefix) for prefix in rule.get("inactive_prefixes", []) if str(prefix)]
    if prefixes:
        parent_text = parent_values.astype(str)
        for prefix in prefixes:
            inactive_mask |= parent_text.str.startswith(prefix)

    return inactive_mask


def _rule_parent_selected(
    parent: str,
    include_terms: Sequence[str] | None,
    exclude_terms: Sequence[str] | None,
) -> bool:
    parent_text = parent.lower()

    if include_terms and not any(term in parent_text for term in include_terms):
        return False

    if exclude_terms and any(term in parent_text for term in exclude_terms):
        return False

    return True


def drop_or_resample_invalid_rows(
    df: pd.DataFrame,
    constraints: dict,
    generator=None,
    target_rows: int | None = None,
    max_attempts: int = 3,
) -> pd.DataFrame:
    repaired = df.copy()
    invalid_mask = _invalid_row_mask(repaired, constraints)
    repaired = repaired.loc[~invalid_mask].reset_index(drop=True)

    if generator is None or target_rows is None or len(repaired) >= target_rows:
        return repaired

    while len(repaired) < target_rows and max_attempts > 0:
        needed = target_rows - len(repaired)
        replacement = generator.sample(needed)
        replacement = repair_dataframe(replacement, constraints)
        replacement = replacement.loc[~_invalid_row_mask(replacement, constraints)].reset_index(drop=True)
        repaired = pd.concat([repaired, replacement], ignore_index=True).head(target_rows)
        max_attempts -= 1

    return repaired


def repair_dataframe(
    df: pd.DataFrame,
    constraints: dict,
    generator=None,
    target_rows: int | None = None,
) -> pd.DataFrame:
    repaired = clip_ranges(df, constraints)
    repaired = normalize_categories(repaired, constraints)
    repaired = enforce_conditional_blanks(repaired, constraints)
    repaired = recompute_derived_fields(repaired, constraints)
    repaired = drop_or_resample_invalid_rows(
        repaired,
        constraints,
        generator=generator,
        target_rows=target_rows,
    )
    return repaired.reset_index(drop=True)


def _invalid_row_mask(df: pd.DataFrame, constraints: dict) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    repair_options = constraints.get("repair_options", {})
    derived_repair_mode = str(repair_options.get("derived_repair_mode", "overwrite")).lower()

    for column, rate in range_violations(df, constraints).items():
        if rate <= 0:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        lower, upper = constraints["ranges"][column]
        mask |= ~(numeric.between(lower, upper) | numeric.isna())

    for column, rate in category_violations(df, constraints).items():
        if rate <= 0:
            continue
        allowed = constraints["allowed_values"][column]
        mask |= ~(df[column].isin(allowed) | df[column].isna())

    for column, rate in derived_field_violations(df, constraints).items():
        if rate <= 0:
            continue
        rule = next(rule for rule in constraints["derived"] if rule["target"] == column)
        kind = str(rule.get("kind", "numeric")).lower()
        if derived_repair_mode == "fill_missing" and kind in {"datetime", "time"}:
            continue
        if derived_repair_mode == "fill_missing_datetime" and kind in {"datetime", "time"}:
            continue
        mask |= derived_field_violation_mask(df, rule)

    return mask
