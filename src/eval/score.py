from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.eval.dependencies import dependency_score
from src.eval.discriminator import compute_discriminator_auc
from src.eval.logic import logic_score
from src.eval.marginals import (
    aggregate_marginal_score,
    jsd_by_categorical_column,
    ks_by_numeric_column,
    tv_by_categorical_column,
    wasserstein_by_numeric_column,
)
from src.eval.privacy import privacy_score
from src.utils.types import RunResult, Schema


OFFICIAL_ORDER_WEIGHTS = {
    "marginal": 0.40,
    "dependency": 0.30,
    "privacy": 0.20,
    "discriminator": 0.10,
}


def compute_total_score(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    schema: Schema,
    constraints: dict,
    weights: dict[str, float],
) -> dict[str, Any]:
    marginal = aggregate_marginal_score(
        {
            "ks": ks_by_numeric_column(real_df, syn_df, schema),
            "wasserstein": wasserstein_by_numeric_column(real_df, syn_df, schema),
            "tv": tv_by_categorical_column(real_df, syn_df, schema),
            "jsd": jsd_by_categorical_column(real_df, syn_df, schema),
        }
    )
    dependency = dependency_score(real_df, syn_df, schema)
    discriminator = compute_discriminator_auc(real_df, syn_df, schema)
    privacy = privacy_score(real_df, syn_df, schema)
    logic = logic_score(syn_df, constraints)

    total_score = float(
        weights.get("marginal", 0.0) * marginal["score"]
        + weights.get("dependency", 0.0) * dependency["score"]
        + weights.get("discriminator", 0.0) * discriminator["score"]
        + weights.get("privacy", 0.0) * privacy["score"]
        + weights.get("logic", 0.0) * logic["score"]
    )

    return {
        "total_score": total_score,
        "marginal": marginal,
        "dependency": dependency,
        "discriminator": discriminator,
        "privacy": privacy,
        "logic": logic,
    }


def compute_official_order_score(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    schema: Schema,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Score in the stated competition priority order.

    This intentionally excludes clinical logic from the scalar score. Missing
    values are treated as observed structural states by the component metrics;
    explicit "Unknown" values remain ordinary category values.
    """
    marginal = aggregate_marginal_score(
        {
            "ks": ks_by_numeric_column(real_df, syn_df, schema),
            "wasserstein": wasserstein_by_numeric_column(real_df, syn_df, schema),
            "tv": tv_by_categorical_column(real_df, syn_df, schema),
            "jsd": jsd_by_categorical_column(real_df, syn_df, schema),
        }
    )
    dependency = dependency_score(real_df, syn_df, schema)
    privacy = privacy_score(real_df, syn_df, schema)
    discriminator = compute_discriminator_auc(real_df, syn_df, schema)

    ordered_weights = _normalized_official_order_weights(weights)
    total_score = official_order_score_from_components(
        marginal=float(marginal["score"]),
        dependency=float(dependency["score"]),
        privacy=float(privacy["score"]),
        discriminator=float(discriminator["score"]),
        weights=ordered_weights,
    )

    return {
        "total_score": total_score,
        "weights": ordered_weights,
        "marginal": marginal,
        "dependency": dependency,
        "privacy": privacy,
        "discriminator": discriminator,
    }


def official_order_score_from_components(
    marginal: float,
    dependency: float,
    privacy: float,
    discriminator: float,
    weights: dict[str, float] | None = None,
) -> float:
    ordered_weights = _normalized_official_order_weights(weights)
    return float(
        ordered_weights["marginal"] * float(marginal)
        + ordered_weights["dependency"] * float(dependency)
        + ordered_weights["privacy"] * float(privacy)
        + ordered_weights["discriminator"] * float(discriminator)
    )


def official_order_score_from_metrics(
    metrics: dict[str, Any],
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    ordered_weights = _normalized_official_order_weights(weights)
    component_scores = {
        "marginal": float(metrics["marginal"]["score"]),
        "dependency": float(metrics["dependency"]["score"]),
        "privacy": float(metrics["privacy"]["score"]),
        "discriminator": float(metrics["discriminator"]["score"]),
    }
    return {
        "total_score": official_order_score_from_components(
            weights=ordered_weights,
            **component_scores,
        ),
        "weights": ordered_weights,
        "component_scores": component_scores,
    }


def compare_runs(run_metrics_list: list[RunResult]) -> list[dict[str, Any]]:
    ranked = sorted(run_metrics_list, key=lambda run: run.total_score, reverse=True)
    return [
        {
            "model_name": run.model_name,
            "total_score": run.total_score,
            "artifact_path": run.artifact_path,
            "notes": run.notes,
        }
        for run in ranked
    ]


def _normalized_official_order_weights(weights: dict[str, float] | None = None) -> dict[str, float]:
    source = OFFICIAL_ORDER_WEIGHTS if weights is None else weights
    resolved = {
        component: max(float(source.get(component, 0.0)), 0.0)
        for component in OFFICIAL_ORDER_WEIGHTS
    }
    total = sum(resolved.values())
    if total <= 0.0:
        return OFFICIAL_ORDER_WEIGHTS.copy()
    return {component: float(weight / total) for component, weight in resolved.items()}
