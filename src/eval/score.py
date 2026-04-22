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

