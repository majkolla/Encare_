from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import pandas as pd

from src.data.mixed import split_mixed_column
from src.data.loader import load_source_csv
from src.data.schema import infer_schema
from src.eval.dependencies import dependency_score
from src.eval.discriminator import compute_discriminator_auc
from src.eval.logic import conditional_blank_violations, logic_score
from src.eval.marginals import (
    aggregate_marginal_score,
    jsd_by_categorical_column,
    ks_by_numeric_column,
    tv_by_categorical_column,
    wasserstein_by_numeric_column,
)
from src.eval.privacy import privacy_score
from src.generate import run_generation
from src.rules.constraints import build_default_constraints
from src.utils.io import ensure_dir, merge_dicts, read_config, write_json, write_markdown
from src.utils.paths import resolve_repo_path

_CANDIDATE_SCORE_COMPONENTS = ("marginal", "dependency", "discriminator", "privacy", "logic")
_DEFAULT_CANDIDATE_SCORE_WEIGHTS = {
    "marginal": 0.35,
    "dependency": 0.35,
    "discriminator": 0.20,
    "privacy": 0.0,
    "logic": 0.10,
}
_LEADERBOARD_SCORE_COMPONENTS = ("marginal", "dependency", "logic", "support", "rare", "conditional", "missingness")
_DEFAULT_LEADERBOARD_SCORE_WEIGHTS = {
    "marginal": 0.25,
    "dependency": 0.40,
    "logic": 0.10,
    "support": 0.15,
    "rare": 0.10,
    "conditional": 0.0,
    "missingness": 0.0,
}


def run_candidate_search(
    model_path: str,
    config_path: str,
    data_path: str,
    output_dir: str,
    prefix: str,
    num_candidates: int,
    seed_start: int,
    reuse_existing: bool = False,
    oversample_factors: Sequence[float] | None = None,
    max_attempts_values: Sequence[int] | None = None,
    privacy_min_distances: Sequence[float] | None = None,
    privacy_min_distance_quantiles: Sequence[float] | None = None,
    candidate_score_weights: dict[str, float] | None = None,
    discriminator_sample_rows: int | None = None,
    privacy_sample_rows: int | None = None,
) -> dict[str, Any]:
    model_path_obj = resolve_repo_path(model_path)
    config_path_obj = resolve_repo_path(config_path)
    data_path_obj = resolve_repo_path(data_path)
    output_dir_obj = ensure_dir(resolve_repo_path(output_dir))

    base_config = read_config(resolve_repo_path("configs/base.yaml"))
    run_config = merge_dicts(base_config, read_config(config_path_obj))
    search_config = run_config.get("candidate_search", {})

    real_df = load_source_csv(data_path_obj)
    schema = infer_schema(real_df)
    real_missingness = real_df.isna().mean()
    constraints = build_default_constraints(
        real_df,
        schema,
        include_conditional_blanks=bool(run_config.get("include_conditional_blanks", False)),
        derived_repair_mode=str(run_config.get("derived_repair_mode", "overwrite")),
    )

    model_name = _resolve_model_name(run_config)
    model_config = _model_config_for_name(run_config, model_name)

    oversample_grid = _resolve_grid_values(
        explicit_values=oversample_factors,
        config_values=search_config.get("oversample_factors"),
        fallback_values=[float(model_config.get("oversample_factor", 1.1))],
        cast=float,
    )
    max_attempts_grid = _resolve_grid_values(
        explicit_values=max_attempts_values,
        config_values=search_config.get("max_attempts_values"),
        fallback_values=[int(model_config.get("max_attempts", 4))],
        cast=int,
    )
    privacy_distance_grid = _resolve_grid_values(
        explicit_values=privacy_min_distances,
        config_values=search_config.get("privacy_min_distances"),
        fallback_values=[float(model_config.get("privacy_min_distance", 0.0))],
        cast=float,
    )
    privacy_distance_quantile_grid = _resolve_grid_values(
        explicit_values=privacy_min_distance_quantiles,
        config_values=search_config.get("privacy_min_distance_quantiles"),
        fallback_values=[float(model_config.get("privacy_min_distance_quantile", 0.0) or 0.0)],
        cast=float,
    )
    proxy_weights = _resolve_candidate_score_weights(
        explicit_weights=candidate_score_weights,
        config_weights=search_config.get("proxy_weights"),
    )
    leaderboard_weights = _resolve_leaderboard_score_weights(search_config.get("leaderboard_proxy_weights"))
    ranking_metric = str(search_config.get("ranking_metric", "proxy")).lower()
    support_max_unique = int(search_config.get("support_max_unique", 12))
    discriminator_sample_rows = int(
        discriminator_sample_rows
        if discriminator_sample_rows is not None
        else search_config.get("discriminator_sample_rows", 2000)
    )
    privacy_sample_rows = int(
        privacy_sample_rows
        if privacy_sample_rows is not None
        else search_config.get("privacy_sample_rows", 2000)
    )

    default_oversample = float(model_config.get("oversample_factor", 1.1))
    default_max_attempts = int(model_config.get("max_attempts", 4))
    default_privacy_distance = float(model_config.get("privacy_min_distance", 0.0))
    default_privacy_distance_quantile = float(model_config.get("privacy_min_distance_quantile", 0.0) or 0.0)

    candidate_rows: list[dict[str, Any]] = []
    for offset in range(num_candidates):
        sample_seed = seed_start + offset
        for oversample_factor, max_attempts, privacy_min_distance, privacy_min_distance_quantile in itertools.product(
            oversample_grid,
            max_attempts_grid,
            privacy_distance_grid,
            privacy_distance_quantile_grid,
        ):
            output_stem = _candidate_output_stem(
                prefix=prefix,
                sample_seed=sample_seed,
                oversample_factor=oversample_factor,
                max_attempts=max_attempts,
                privacy_min_distance=privacy_min_distance,
                privacy_min_distance_quantile=privacy_min_distance_quantile,
                default_oversample=default_oversample,
                default_max_attempts=default_max_attempts,
                default_privacy_min_distance=default_privacy_distance,
                default_privacy_min_distance_quantile=default_privacy_distance_quantile,
            )
            output_path = output_dir_obj / f"{output_stem}.csv"
            generation_override = _build_generation_override(
                model_name=model_name,
                oversample_factor=oversample_factor,
                max_attempts=max_attempts,
                privacy_min_distance=privacy_min_distance,
                privacy_min_distance_quantile=privacy_min_distance_quantile,
            )

            if reuse_existing and output_path.exists():
                generation_result = {
                    "output_path": str(output_path),
                    "note_path": str(output_path.with_suffix(".md")),
                    "errors": [],
                }
            else:
                generation_result = run_generation(
                    model_path=str(model_path_obj),
                    config_path=str(config_path_obj),
                    data_path=str(data_path_obj),
                    output_path=str(output_path),
                    sample_seed=sample_seed,
                    config_override=generation_override,
                )

            candidate_df = pd.read_csv(output_path, low_memory=False)
            support_summary = _mixed_support_summary(real_df, candidate_df, schema)
            missingness_summary = _missingness_summary(real_missingness, candidate_df)
            proxy_summary = _candidate_proxy_summary(
                real_df=real_df,
                syn_df=candidate_df,
                schema=schema,
                constraints=constraints,
                weights=proxy_weights,
                discriminator_sample_rows=discriminator_sample_rows,
                privacy_sample_rows=privacy_sample_rows,
                required_components=("marginal", "dependency", "logic"),
            )
            support_feature_summary = _support_feature_summary(
                real_df=real_df,
                syn_df=candidate_df,
                schema=schema,
                max_unique=support_max_unique,
            )
            conditional_blank_summary = _conditional_blank_summary(
                real_df=real_df,
                syn_df=candidate_df,
                constraints=constraints,
            )
            leaderboard_summary = _leaderboard_proxy_summary(
                proxy_component_scores=proxy_summary["component_scores"],
                support_feature_summary=support_feature_summary,
                conditional_blank_summary=conditional_blank_summary,
                missingness_summary=missingness_summary,
                weights=leaderboard_weights,
            )

            candidate_rows.append(
                {
                    "sample_seed": sample_seed,
                    "oversample_factor": oversample_factor,
                    "max_attempts": max_attempts,
                    "privacy_min_distance": privacy_min_distance,
                    "privacy_min_distance_quantile": privacy_min_distance_quantile,
                    "output_path": str(output_path),
                    "note_path": generation_result["note_path"],
                    "sha256": _sha256_file(output_path),
                    "errors": generation_result["errors"],
                    "proxy_total_score": proxy_summary["total_score"],
                    "proxy_component_scores": proxy_summary["component_scores"],
                    "leaderboard_total_score": leaderboard_summary["total_score"],
                    "leaderboard_component_scores": leaderboard_summary["component_scores"],
                    "mixed_extra_support_total": support_summary["extra_support_total"],
                    "mixed_missing_support_total": support_summary["missing_support_total"],
                    "mixed_support_by_column": support_summary["by_column"],
                    "support_feature_summary": support_feature_summary,
                    "conditional_blank_score": conditional_blank_summary["score"],
                    "conditional_blank_mean_abs_gap": conditional_blank_summary["mean_abs_gap"],
                    "conditional_blank_real_rate": conditional_blank_summary["real_overall_rate"],
                    "conditional_blank_syn_rate": conditional_blank_summary["syn_overall_rate"],
                    "conditional_blank_top_rule_gaps": conditional_blank_summary["top_rule_gaps"],
                    "duplicate_rows": int(candidate_df.duplicated().sum()),
                    "mean_abs_missingness_delta": missingness_summary["mean_abs_delta"],
                    "max_abs_missingness_delta": missingness_summary["max_abs_delta"],
                    "top_missingness_drift_columns": missingness_summary["top_columns"],
                }
            )

    if ranking_metric not in {"proxy", "leaderboard_proxy"}:
        raise ValueError(f"Unknown ranking metric: {ranking_metric}")

    ranked_candidates = sorted(
        candidate_rows,
        key=lambda row: (
            -float(
                row["leaderboard_total_score"]
                if ranking_metric == "leaderboard_proxy"
                else row["proxy_total_score"]
            ),
            -float(row.get("conditional_blank_score", 0.0)),
            -float(row["leaderboard_component_scores"].get("support", 0.0)),
            -float(row["proxy_component_scores"].get("dependency", 0.0)),
            int(row["mixed_extra_support_total"]),
            float(row["max_abs_missingness_delta"]),
            float(row["mean_abs_missingness_delta"]),
            int(row["mixed_missing_support_total"]),
            int(row["duplicate_rows"]),
        ),
    )
    payload = {
        "model_path": str(model_path_obj),
        "config_path": str(config_path_obj),
        "data_path": str(data_path_obj),
        "output_dir": str(output_dir_obj),
        "prefix": prefix,
        "num_candidates": num_candidates,
        "seed_start": seed_start,
        "reuse_existing": reuse_existing,
        "candidate_score_weights": proxy_weights,
        "leaderboard_proxy_weights": leaderboard_weights,
        "ranking_metric": ranking_metric,
        "search_grid": {
            "oversample_factors": oversample_grid,
            "max_attempts_values": max_attempts_grid,
            "privacy_min_distances": privacy_distance_grid,
            "privacy_min_distance_quantiles": privacy_distance_quantile_grid,
            "discriminator_sample_rows": discriminator_sample_rows,
            "privacy_sample_rows": privacy_sample_rows,
            "support_max_unique": support_max_unique,
        },
        "ranked_candidates": ranked_candidates,
    }
    write_json(_to_json_safe(payload), output_dir_obj / f"{prefix}_search_summary.json")
    write_markdown(
        _render_search_markdown(payload),
        output_dir_obj / f"{prefix}_search_summary.md",
    )
    return payload


def _candidate_proxy_summary(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    schema,
    constraints: dict[str, Any],
    weights: dict[str, float],
    discriminator_sample_rows: int,
    privacy_sample_rows: int,
    required_components: Sequence[str] | None = None,
) -> dict[str, Any]:
    component_scores: dict[str, float] = {}
    required = set(required_components or [])
    positive_weights = {
        component: float(weight)
        for component, weight in weights.items()
        if component in _CANDIDATE_SCORE_COMPONENTS and float(weight) > 0.0
    }

    if positive_weights.get("marginal", 0.0) > 0.0 or "marginal" in required:
        marginal = aggregate_marginal_score(
            {
                "ks": ks_by_numeric_column(real_df, syn_df, schema),
                "wasserstein": wasserstein_by_numeric_column(real_df, syn_df, schema),
                "tv": tv_by_categorical_column(real_df, syn_df, schema),
                "jsd": jsd_by_categorical_column(real_df, syn_df, schema),
            }
        )
        component_scores["marginal"] = float(marginal["score"])

    if positive_weights.get("dependency", 0.0) > 0.0 or "dependency" in required:
        dependency = dependency_score(real_df, syn_df, schema)
        component_scores["dependency"] = float(dependency["score"])

    if positive_weights.get("logic", 0.0) > 0.0 or "logic" in required:
        logic = logic_score(syn_df, constraints)
        component_scores["logic"] = float(logic["score"])

    if positive_weights.get("discriminator", 0.0) > 0.0:
        real_disc, syn_disc = _sample_pair(real_df, syn_df, discriminator_sample_rows)
        discriminator = compute_discriminator_auc(real_disc, syn_disc, schema)
        component_scores["discriminator"] = float(discriminator["score"])

    if positive_weights.get("privacy", 0.0) > 0.0:
        real_priv, syn_priv = _sample_pair(real_df, syn_df, privacy_sample_rows)
        privacy = privacy_score(real_priv, syn_priv, schema)
        component_scores["privacy"] = float(privacy["score"])

    total_weight = sum(
        float(weight)
        for component, weight in positive_weights.items()
        if component in component_scores
    )
    total_score = 0.0
    if total_weight > 0.0:
        total_score = sum(
            float(positive_weights[component]) * float(score)
            for component, score in component_scores.items()
        ) / total_weight

    return {
        "total_score": float(total_score),
        "component_scores": component_scores,
    }


def _leaderboard_proxy_summary(
    proxy_component_scores: dict[str, float],
    support_feature_summary: dict[str, Any],
    conditional_blank_summary: dict[str, Any],
    missingness_summary: dict[str, Any],
    weights: dict[str, float],
) -> dict[str, Any]:
    component_scores = {
        "marginal": float(proxy_component_scores.get("marginal", 0.0)),
        "dependency": float(proxy_component_scores.get("dependency", 0.0)),
        "logic": float(proxy_component_scores.get("logic", 0.0)),
        "support": float(support_feature_summary.get("support_score", 0.0)),
        "rare": float(support_feature_summary.get("rare_score", 0.0)),
        "conditional": float(conditional_blank_summary.get("score", 1.0)),
        "missingness": _missingness_stability_score(missingness_summary),
    }
    total_weight = sum(float(weights.get(component, 0.0)) for component in _LEADERBOARD_SCORE_COMPONENTS)
    if total_weight <= 0.0:
        return {"total_score": 0.0, "component_scores": component_scores}

    total_score = sum(
        float(weights.get(component, 0.0)) * float(component_scores.get(component, 0.0))
        for component in _LEADERBOARD_SCORE_COMPONENTS
    ) / total_weight
    return {
        "total_score": float(total_score),
        "component_scores": component_scores,
    }


def _sample_pair(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    max_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if max_rows <= 0:
        return real_df, syn_df
    real_sample = real_df.sample(min(len(real_df), max_rows), random_state=42)
    syn_sample = syn_df.sample(min(len(syn_df), max_rows), random_state=42)
    return real_sample.reset_index(drop=True), syn_sample.reset_index(drop=True)


def _build_generation_override(
    model_name: str,
    oversample_factor: float,
    max_attempts: int,
    privacy_min_distance: float,
    privacy_min_distance_quantile: float,
) -> dict[str, Any]:
    model_override: dict[str, Any] = {
        "oversample_factor": float(oversample_factor),
        "max_attempts": int(max_attempts),
        "privacy_min_distance": float(privacy_min_distance),
    }
    if float(privacy_min_distance_quantile) > 0.0:
        model_override["privacy_min_distance_quantile"] = float(privacy_min_distance_quantile)

    return {
        model_name: model_override
    }


def _candidate_output_stem(
    prefix: str,
    sample_seed: int,
    oversample_factor: float,
    max_attempts: int,
    privacy_min_distance: float,
    privacy_min_distance_quantile: float,
    default_oversample: float,
    default_max_attempts: int,
    default_privacy_min_distance: float,
    default_privacy_min_distance_quantile: float,
) -> str:
    suffixes: list[str] = []
    if not _float_close(oversample_factor, default_oversample):
        suffixes.append(f"os{_compact_value(oversample_factor)}")
    if max_attempts != default_max_attempts:
        suffixes.append(f"ma{max_attempts}")
    if not _float_close(privacy_min_distance, default_privacy_min_distance):
        suffixes.append(f"pd{_compact_value(privacy_min_distance)}")
    if not _float_close(privacy_min_distance_quantile, default_privacy_min_distance_quantile):
        suffixes.append(f"pq{_compact_value(privacy_min_distance_quantile)}")

    parts = [prefix, str(sample_seed), *suffixes]
    return "_".join(parts)


def _resolve_model_name(run_config: dict[str, Any]) -> str:
    configured_model = run_config.get("model")
    if configured_model:
        return str(configured_model)

    configured_models = run_config.get("models", [])
    if isinstance(configured_models, list) and len(configured_models) == 1:
        return str(configured_models[0])

    raise ValueError("Candidate search requires a config with one explicit model name.")


def _model_config_for_name(run_config: dict[str, Any], model_name: str) -> dict[str, Any]:
    model_config = run_config.get(model_name, {})
    if isinstance(model_config, dict):
        return merge_dicts(run_config, model_config)
    return dict(run_config)


def _resolve_grid_values(
    explicit_values: Sequence[Any] | None,
    config_values: Any,
    fallback_values: Sequence[Any],
    cast: Callable[[Any], Any],
) -> list[Any]:
    source_values: Iterable[Any]
    if explicit_values is not None:
        source_values = explicit_values
    elif isinstance(config_values, list) and config_values:
        source_values = config_values
    else:
        source_values = fallback_values

    resolved: list[Any] = []
    for value in source_values:
        cast_value = cast(value)
        if cast_value not in resolved:
            resolved.append(cast_value)
    return resolved


def _resolve_candidate_score_weights(
    explicit_weights: dict[str, float] | None,
    config_weights: Any,
) -> dict[str, float]:
    source = explicit_weights if explicit_weights is not None else config_weights
    weights = dict(_DEFAULT_CANDIDATE_SCORE_WEIGHTS)

    if isinstance(source, dict):
        for component in _CANDIDATE_SCORE_COMPONENTS:
            if component in source:
                weights[component] = float(source[component])
    return weights


def _resolve_leaderboard_score_weights(config_weights: Any) -> dict[str, float]:
    weights = dict(_DEFAULT_LEADERBOARD_SCORE_WEIGHTS)
    if isinstance(config_weights, dict):
        for component in _LEADERBOARD_SCORE_COMPONENTS:
            if component in config_weights:
                weights[component] = float(config_weights[component])
    return weights


def _mixed_support_summary(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    schema,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for column in schema.columns:
        if column.mixed_value_kind is None:
            continue
        real_support = set(real_df[column.name].dropna().astype(str).unique())
        syn_support = set(syn_df[column.name].dropna().astype(str).unique())
        rows.append(
            {
                "column": column.name,
                "extra_support": len(syn_support - real_support),
                "missing_support": len(real_support - syn_support),
                "synthetic_unique": len(syn_support),
            }
        )

    return {
        "extra_support_total": sum(row["extra_support"] for row in rows),
        "missing_support_total": sum(row["missing_support"] for row in rows),
        "by_column": rows,
    }


def _support_feature_summary(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    schema,
    max_unique: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for feature_name, feature_kind, real_feature, syn_feature in _iter_support_feature_pairs(
        real_df=real_df,
        syn_df=syn_df,
        schema=schema,
        max_unique=max_unique,
    ):
        real_counts = real_feature.value_counts(normalize=True, sort=False)
        syn_counts = syn_feature.value_counts(normalize=True, sort=False)
        index = real_counts.index.union(syn_counts.index)
        real_dist = real_counts.reindex(index, fill_value=0.0)
        syn_dist = syn_counts.reindex(index, fill_value=0.0)

        real_support = real_dist[real_dist > 0.0]
        shared_support = ((real_dist > 0.0) & (syn_dist > 0.0)).sum()
        support_recall = float(shared_support / max(len(real_support), 1))
        weighted_drift = _weighted_distribution_drift(real_dist, syn_dist)

        rows.append(
            {
                "feature": feature_name,
                "kind": feature_kind,
                "support_recall": support_recall,
                "weighted_drift": weighted_drift,
            }
        )

    if not rows:
        return {
            "support_score": 0.0,
            "rare_score": 0.0,
            "low_card_support_score": 0.0,
            "mixed_state_support_score": 0.0,
            "top_drift_features": [],
        }

    low_card_rows = [row for row in rows if row["kind"] == "low_card"]
    mixed_rows = [row for row in rows if row["kind"] == "mixed_state"]
    support_score = float(pd.Series([row["support_recall"] for row in rows], dtype="float64").mean())
    rare_score = float(
        pd.Series([max(0.0, 1.0 - row["weighted_drift"]) for row in rows], dtype="float64").mean()
    )

    return {
        "support_score": support_score,
        "rare_score": rare_score,
        "low_card_support_score": float(
            pd.Series([row["support_recall"] for row in low_card_rows], dtype="float64").mean()
        ) if low_card_rows else 0.0,
        "mixed_state_support_score": float(
            pd.Series([row["support_recall"] for row in mixed_rows], dtype="float64").mean()
        ) if mixed_rows else 0.0,
        "top_drift_features": sorted(rows, key=lambda row: float(row["weighted_drift"]), reverse=True)[:10],
    }


def _iter_support_feature_pairs(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    schema,
    max_unique: int,
):
    for column in schema.columns:
        column_name = column.name
        if column_name not in syn_df.columns:
            continue

        if column.mixed_value_kind is not None:
            _, _, real_states = split_mixed_column(
                column_name=column_name,
                series=real_df[column_name],
                source_kind=column.mixed_value_kind,
            )
            _, _, syn_states = split_mixed_column(
                column_name=column_name,
                series=syn_df[column_name],
                source_kind=column.mixed_value_kind,
            )
            yield (
                f"{column_name}__state",
                "mixed_state",
                _normalized_support_feature(real_states),
                _normalized_support_feature(syn_states),
            )
            continue

        if column.kind in {"categorical", "binary", "id_like"} and column.unique_count <= max_unique:
            yield (
                column_name,
                "low_card",
                _normalized_support_feature(real_df[column_name]),
                _normalized_support_feature(syn_df[column_name]),
            )


def _normalized_support_feature(series: pd.Series) -> pd.Series:
    return series.astype("object").where(series.notna(), "__MISSING__")


def _weighted_distribution_drift(real_dist: pd.Series, syn_dist: pd.Series) -> float:
    if real_dist.empty and syn_dist.empty:
        return 0.0

    min_positive = float(real_dist[real_dist > 0.0].min()) if (real_dist > 0.0).any() else 1.0
    baseline = max(min_positive, 1e-9)
    weights = 1.0 / (real_dist.clip(lower=baseline) ** 0.5)
    drift = 0.5 * float((weights * (real_dist - syn_dist).abs()).sum() / weights.sum())
    return float(min(max(drift, 0.0), 1.0))


def _missingness_summary(
    real_missingness: pd.Series,
    syn_df: pd.DataFrame,
) -> dict[str, Any]:
    deltas = (syn_df.isna().mean() - real_missingness).abs().sort_values(ascending=False)
    top_columns = [
        {
            "column": str(column),
            "abs_missingness_delta": float(value),
        }
        for column, value in deltas.head(10).items()
    ]
    return {
        "mean_abs_delta": float(deltas.mean()),
        "max_abs_delta": float(deltas.max()),
        "top_columns": top_columns,
    }


def _missingness_stability_score(summary: dict[str, Any]) -> float:
    mean_abs = float(summary.get("mean_abs_delta", 0.0))
    max_abs = float(summary.get("max_abs_delta", 0.0))
    mean_score = 1.0 / (1.0 + 25.0 * mean_abs)
    max_score = 1.0 / (1.0 + 10.0 * max_abs)
    return float((mean_score + max_score) / 2.0)


def _conditional_blank_summary(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    constraints: dict[str, Any],
) -> dict[str, Any]:
    real_rates = conditional_blank_violations(real_df, constraints)
    syn_rates = conditional_blank_violations(syn_df, constraints)
    rule_names = sorted(set(real_rates) | set(syn_rates))
    if not rule_names:
        return {
            "score": 1.0,
            "mean_abs_gap": 0.0,
            "real_overall_rate": 0.0,
            "syn_overall_rate": 0.0,
            "top_rule_gaps": [],
        }

    rows: list[dict[str, float | str]] = []
    for rule_name in rule_names:
        real_rate = float(real_rates.get(rule_name, 0.0))
        syn_rate = float(syn_rates.get(rule_name, 0.0))
        rows.append(
            {
                "rule": str(rule_name),
                "real_rate": real_rate,
                "syn_rate": syn_rate,
                "abs_gap": abs(real_rate - syn_rate),
            }
        )

    mean_abs_gap = float(pd.Series([row["abs_gap"] for row in rows], dtype="float64").mean())
    real_overall_rate = float(pd.Series([row["real_rate"] for row in rows], dtype="float64").mean())
    syn_overall_rate = float(pd.Series([row["syn_rate"] for row in rows], dtype="float64").mean())
    mean_score = 1.0 / (1.0 + 10.0 * mean_abs_gap)
    overall_score = 1.0 / (1.0 + 10.0 * abs(syn_overall_rate - real_overall_rate))

    return {
        "score": float((mean_score + overall_score) / 2.0),
        "mean_abs_gap": mean_abs_gap,
        "real_overall_rate": real_overall_rate,
        "syn_overall_rate": syn_overall_rate,
        "top_rule_gaps": sorted(rows, key=lambda row: float(row["abs_gap"]), reverse=True)[:10],
    }


def _to_json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        pass

    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _render_search_markdown(payload: dict[str, Any]) -> str:
    lines = ["# Candidate Search", ""]
    lines.append(f"- Model artifact: {payload['model_path']}")
    lines.append(f"- Config: {payload['config_path']}")
    lines.append(f"- Candidate count: {len(payload['ranked_candidates'])}")
    lines.append(f"- Seed count: {payload['num_candidates']}")
    lines.append(f"- Seed start: {payload['seed_start']}")
    lines.append(f"- Reused existing files: {payload['reuse_existing']}")
    lines.append(f"- Ranking metric: {payload['ranking_metric']}")
    lines.append(f"- Proxy weights: {payload['candidate_score_weights']}")
    lines.append(f"- Leaderboard proxy weights: {payload['leaderboard_proxy_weights']}")
    lines.append(f"- Search grid: {payload['search_grid']}")
    lines.extend(["", "## Ranked Candidates", ""])
    for row in payload["ranked_candidates"]:
        component_scores = row.get("proxy_component_scores", {})
        component_summary = ", ".join(
            f"{name}={score:.4f}"
            for name, score in component_scores.items()
        )
        leaderboard_component_scores = row.get("leaderboard_component_scores", {})
        leaderboard_component_summary = ", ".join(
            f"{name}={score:.4f}"
            for name, score in leaderboard_component_scores.items()
        )
        lines.append(
            f"- seed={row['sample_seed']}, "
            f"proxy_score={row['proxy_total_score']:.4f}, "
            f"leaderboard_score={row['leaderboard_total_score']:.4f}, "
            f"oversample={row['oversample_factor']}, "
            f"max_attempts={row['max_attempts']}, "
            f"privacy_min_distance={row['privacy_min_distance']}, "
            f"privacy_min_distance_quantile={row['privacy_min_distance_quantile']}, "
            f"sha256={str(row['sha256'])[:16]}, "
            f"conditional_score={row.get('conditional_blank_score', 1.0):.4f}, "
            f"conditional_gap={row.get('conditional_blank_mean_abs_gap', 0.0):.4f}, "
            f"extra_support={row['mixed_extra_support_total']}, "
            f"max_missingness_delta={row['max_abs_missingness_delta']:.4f}, "
            f"mean_missingness_delta={row['mean_abs_missingness_delta']:.4f}, "
            f"missing_support={row['mixed_missing_support_total']}, "
            f"duplicates={row['duplicate_rows']}, "
            f"components=[{component_summary}], "
            f"leaderboard_components=[{leaderboard_component_summary}], "
            f"file={Path(row['output_path']).name}"
        )
    return "\n".join(lines) + "\n"


def _parse_csv_list(raw: str | None, cast: Callable[[str], Any]) -> list[Any] | None:
    if raw is None:
        return None
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        return None
    return [cast(part) for part in parts]


def _compact_value(value: float | int) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _float_close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def _sha256_file(path: str | Path) -> str:
    file_path = Path(path)
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and summarize multiple candidate outputs from one saved artifact.")
    parser.add_argument("--model-path", required=True, help="Path to a saved model.pkl artifact.")
    parser.add_argument("--config", required=True, help="Config path used to resolve repair/privacy settings.")
    parser.add_argument("--data", default="data/data.csv", help="Path to the source CSV.")
    parser.add_argument("--output-dir", default="data/outputs", help="Directory where candidate CSVs will be written.")
    parser.add_argument("--prefix", default="candidate", help="Filename prefix for generated candidates.")
    parser.add_argument("--num-candidates", type=int, default=4, help="How many candidate seeds to generate.")
    parser.add_argument("--seed-start", type=int, default=100, help="Starting sample seed for candidate generation.")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse already-generated candidate CSVs instead of regenerating them.",
    )
    parser.add_argument(
        "--oversample-factors",
        help="Optional comma-separated oversample-factor grid. Example: 1.05,1.1,1.2",
    )
    parser.add_argument(
        "--max-attempts-values",
        help="Optional comma-separated max-attempts grid. Example: 4,6,8",
    )
    parser.add_argument(
        "--privacy-min-distances",
        help="Optional comma-separated privacy-min-distance grid. Example: 0.0,0.02,0.05",
    )
    parser.add_argument(
        "--privacy-min-distance-quantiles",
        help="Optional comma-separated privacy-min-distance quantile grid. Example: 0.0,0.01,0.05",
    )
    parser.add_argument(
        "--discriminator-sample-rows",
        type=int,
        default=None,
        help="How many rows per table to use when scoring discriminator drift.",
    )
    parser.add_argument(
        "--privacy-sample-rows",
        type=int,
        default=None,
        help="How many rows per table to use when scoring privacy drift.",
    )
    args = parser.parse_args()

    result = run_candidate_search(
        model_path=args.model_path,
        config_path=args.config,
        data_path=args.data,
        output_dir=args.output_dir,
        prefix=args.prefix,
        num_candidates=args.num_candidates,
        seed_start=args.seed_start,
        reuse_existing=args.reuse_existing,
        oversample_factors=_parse_csv_list(args.oversample_factors, float),
        max_attempts_values=_parse_csv_list(args.max_attempts_values, int),
        privacy_min_distances=_parse_csv_list(args.privacy_min_distances, float),
        privacy_min_distance_quantiles=_parse_csv_list(args.privacy_min_distance_quantiles, float),
        discriminator_sample_rows=args.discriminator_sample_rows,
        privacy_sample_rows=args.privacy_sample_rows,
    )
    print(result)


if __name__ == "__main__":
    main()
