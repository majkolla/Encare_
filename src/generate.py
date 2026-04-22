from __future__ import annotations

import argparse
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.mixed import split_mixed_column
from src.data.loader import load_source_csv
from src.data.schema import infer_schema
from src.eval.privacy import filter_privacy_violations
from src.models.base import BaseSynthesizer
from src.rules.repair import repair_dataframe
from src.rules.constraints import build_default_constraints
from src.submit import save_validated_submission
from src.utils.io import ensure_dir, merge_dicts, read_config, write_markdown
from src.utils.paths import resolve_repo_path
from src.utils.types import Schema


def generate_synthetic_dataset(
    model: BaseSynthesizer,
    real_df: pd.DataFrame,
    schema: Schema,
    constraints: dict,
    n_rows: int,
    repair: bool = True,
    privacy_filter: bool = True,
    privacy_min_distance: float = 0.0,
    privacy_min_distance_quantile: float | None = None,
    oversample_factor: float = 1.1,
    max_attempts: int = 4,
    selection_strategy: str = "head",
    selection_max_unique: int = 12,
    selection_missingness_weight: float = 0.35,
    selection_deficit_bias: float = 1.5,
    selection_seed: int | None = None,
) -> pd.DataFrame:
    target_rows = n_rows
    synthetic = pd.DataFrame(columns=schema.column_order)

    for _ in range(max_attempts):
        needed = max(target_rows - len(synthetic), 0)
        if needed == 0:
            break

        batch_size = max(int(round(needed * oversample_factor)), needed)
        batch = model.sample(batch_size)

        if repair:
            batch = repair_dataframe(batch, constraints, generator=model, target_rows=None)

        if privacy_filter:
            batch = filter_privacy_violations(
                real_df,
                batch,
                schema,
                min_distance=privacy_min_distance,
                min_distance_quantile=privacy_min_distance_quantile,
            )

        if synthetic.empty:
            synthetic = batch.reset_index(drop=True)
        else:
            synthetic = pd.concat([synthetic, batch], ignore_index=True)
        synthetic = synthetic.drop_duplicates().reset_index(drop=True)
        synthetic = _select_target_rows(
            candidate_df=synthetic,
            real_df=real_df,
            schema=schema,
            target_rows=target_rows,
            strategy=selection_strategy,
            max_unique=selection_max_unique,
            missingness_weight=selection_missingness_weight,
            deficit_bias=selection_deficit_bias,
            random_seed=selection_seed,
        )

    return synthetic.reindex(columns=schema.column_order).head(target_rows).reset_index(drop=True)


def generate_and_save_submission(
    model: BaseSynthesizer,
    real_df: pd.DataFrame,
    schema: Schema,
    constraints: dict,
    output_path: str | Path,
    n_rows: int | None = None,
    repair: bool = True,
    privacy_filter: bool = True,
    privacy_min_distance: float = 0.0,
    privacy_min_distance_quantile: float | None = None,
    oversample_factor: float = 1.1,
    max_attempts: int = 4,
    selection_strategy: str = "head",
    selection_max_unique: int = 12,
    selection_missingness_weight: float = 0.35,
    selection_deficit_bias: float = 1.5,
    selection_seed: int | None = None,
) -> tuple[Path, list[str]]:
    ensure_dir(Path(output_path).parent)
    synthetic = generate_synthetic_dataset(
        model=model,
        real_df=real_df,
        schema=schema,
        constraints=constraints,
        n_rows=n_rows or len(real_df),
        repair=repair,
        privacy_filter=privacy_filter,
        privacy_min_distance=privacy_min_distance,
        privacy_min_distance_quantile=privacy_min_distance_quantile,
        oversample_factor=oversample_factor,
        max_attempts=max_attempts,
        selection_strategy=selection_strategy,
        selection_max_unique=selection_max_unique,
        selection_missingness_weight=selection_missingness_weight,
        selection_deficit_bias=selection_deficit_bias,
        selection_seed=selection_seed,
    )
    return save_validated_submission(synthetic, real_df, schema, output_path)


def write_output_notes(
    output_path: str | Path,
    metadata: dict[str, object],
) -> Path:
    output_path_obj = Path(output_path)
    note_path = output_path_obj.with_suffix(".md")
    lines = [f"# {output_path_obj.name}", ""]

    summary_fields = [
        ("Model", metadata.get("model_name")),
        ("Model artifact", metadata.get("model_path")),
        ("Config", metadata.get("config_path")),
        ("Source data", metadata.get("data_path")),
        ("SHA-256", metadata.get("sha256")),
        ("Rows", metadata.get("n_rows")),
        ("Sample seed", metadata.get("sample_seed")),
        ("Repair", metadata.get("repair")),
        ("Privacy filter", metadata.get("privacy_filter")),
        ("Privacy minimum distance", metadata.get("privacy_min_distance")),
        ("Privacy min distance quantile", metadata.get("privacy_min_distance_quantile")),
        ("Mixed column strategy", metadata.get("mixed_column_strategy")),
        ("Conditional blank rules", metadata.get("include_conditional_blanks")),
        ("Derived repair mode", metadata.get("derived_repair_mode")),
        ("Oversample factor", metadata.get("oversample_factor")),
        ("Max attempts", metadata.get("max_attempts")),
        ("Row selection strategy", metadata.get("selection_strategy")),
        ("Row selection max unique", metadata.get("selection_max_unique")),
        ("Row selection missingness weight", metadata.get("selection_missingness_weight")),
        ("Row selection deficit bias", metadata.get("selection_deficit_bias")),
    ]
    for label, value in summary_fields:
        if value is None:
            continue
        lines.append(f"- {label}: {value}")

    notes = metadata.get("notes")
    if isinstance(notes, list) and notes:
        lines.extend(["", "## Notes", ""])
        for note in notes:
            lines.append(f"- {note}")

    return write_markdown("\n".join(lines) + "\n", note_path)


def run_generation(
    model_path: str,
    config_path: str,
    data_path: str,
    output_path: str | None = None,
    n_rows: int | None = None,
    sample_seed: int | None = None,
    config_override: dict | None = None,
) -> dict:
    model_path_obj = resolve_repo_path(model_path)
    config_path_obj = resolve_repo_path(config_path)
    data_path_obj = resolve_repo_path(data_path)

    base_config = read_config(resolve_repo_path("configs/base.yaml"))
    run_config = merge_dicts(base_config, read_config(config_path_obj))
    if config_override:
        run_config = merge_dicts(run_config, config_override)

    real_df = load_source_csv(data_path_obj)
    schema = infer_schema(real_df)
    constraints = build_default_constraints(
        real_df,
        schema,
        include_conditional_blanks=bool(run_config.get("include_conditional_blanks", False)),
        derived_repair_mode=str(run_config.get("derived_repair_mode", "overwrite")),
    )

    model = BaseSynthesizer.load(model_path_obj)
    if sample_seed is not None:
        _reseed_model(model, sample_seed)
    model_name = model.__class__.__name__.replace("Synthesizer", "").lower()
    model_config = _model_config_for_name(run_config, run_config.get("model", model_name))

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path_obj = resolve_repo_path(
            Path(run_config.get("output_dir", "data/outputs")) / f"{model_name}_submission_{timestamp}.csv"
        )
    else:
        output_path_obj = resolve_repo_path(output_path)

    saved_path, errors = generate_and_save_submission(
        model=model,
        real_df=real_df,
        schema=schema,
        constraints=constraints,
        output_path=output_path_obj,
        n_rows=n_rows or max(len(real_df), int(len(real_df) * run_config.get("n_rows_multiplier", 1.0))),
        repair=model_config.get("repair", False),
        privacy_filter=model_config.get("privacy_filter", False),
        privacy_min_distance=model_config.get("privacy_min_distance", 0.0),
        privacy_min_distance_quantile=model_config.get("privacy_min_distance_quantile"),
        oversample_factor=float(model_config.get("oversample_factor", 1.1)),
        max_attempts=int(model_config.get("max_attempts", 4)),
        selection_strategy=str(model_config.get("selection_strategy", "head")),
        selection_max_unique=int(model_config.get("selection_max_unique", 12)),
        selection_missingness_weight=float(model_config.get("selection_missingness_weight", 0.35)),
        selection_deficit_bias=float(model_config.get("selection_deficit_bias", 1.5)),
        selection_seed=sample_seed,
    )
    note_path = write_output_notes(
        output_path=saved_path,
        metadata={
            "model_name": run_config.get("model", model_name),
            "model_path": str(model_path_obj),
            "config_path": str(config_path_obj),
            "data_path": str(data_path_obj),
            "sha256": _sha256_file(saved_path),
            "n_rows": n_rows or max(len(real_df), int(len(real_df) * run_config.get("n_rows_multiplier", 1.0))),
            "repair": model_config.get("repair", False),
            "privacy_filter": model_config.get("privacy_filter", False),
            "privacy_min_distance": model_config.get("privacy_min_distance", 0.0),
            "privacy_min_distance_quantile": model_config.get("privacy_min_distance_quantile"),
            "mixed_column_strategy": model_config.get("mixed_column_strategy"),
            "sample_seed": sample_seed,
            "include_conditional_blanks": bool(run_config.get("include_conditional_blanks", False)),
            "derived_repair_mode": str(run_config.get("derived_repair_mode", "overwrite")),
            "oversample_factor": float(model_config.get("oversample_factor", 1.1)),
            "max_attempts": int(model_config.get("max_attempts", 4)),
            "selection_strategy": str(model_config.get("selection_strategy", "head")),
            "selection_max_unique": int(model_config.get("selection_max_unique", 12)),
            "selection_missingness_weight": float(model_config.get("selection_missingness_weight", 0.35)),
            "selection_deficit_bias": float(model_config.get("selection_deficit_bias", 1.5)),
            "notes": [
                "Generated via `python -m src.generate` from a saved model artifact.",
                "This sidecar is intended to track how the CSV was produced.",
                *(
                    [f"Generation overrides: {config_override}"]
                    if config_override
                    else []
                ),
            ],
        },
    )

    return {
        "model_path": str(model_path_obj),
        "output_path": str(saved_path),
        "note_path": str(note_path),
        "sha256": _sha256_file(saved_path),
        "errors": errors,
    }


def _select_target_rows(
    candidate_df: pd.DataFrame,
    real_df: pd.DataFrame,
    schema: Schema,
    target_rows: int,
    strategy: str,
    max_unique: int,
    missingness_weight: float,
    deficit_bias: float,
    random_seed: int | None,
) -> pd.DataFrame:
    if len(candidate_df) <= target_rows:
        return candidate_df.head(target_rows).reset_index(drop=True)

    normalized_strategy = str(strategy or "head").lower()
    if normalized_strategy == "head":
        return candidate_df.head(target_rows).reset_index(drop=True)
    if normalized_strategy != "balanced":
        raise ValueError(f"Unknown selection strategy: {strategy}")

    pool = candidate_df.reset_index(drop=True)
    excess = len(pool) - target_rows
    row_scores = np.zeros(len(pool), dtype=float)
    feature_count = 0
    protected_feature_series: dict[str, pd.Series] = {}

    for feature_name, real_feature, pool_feature, weight in _iter_balancing_features(
        real_df=real_df,
        candidate_df=pool,
        schema=schema,
        max_unique=max_unique,
        missingness_weight=missingness_weight,
    ):
        real_dist = real_feature.value_counts(normalize=True, sort=False)
        pool_dist = pool_feature.value_counts(normalize=True, sort=False)
        if pool_dist.empty:
            continue

        index = real_dist.index.union(pool_dist.index)
        real_dist = real_dist.reindex(index, fill_value=0.0)
        pool_dist = pool_dist.reindex(index, fill_value=0.0)

        surplus = ((pool_dist - real_dist).clip(lower=0.0) / pool_dist.replace(0.0, np.nan)).fillna(0.0)
        deficit = ((real_dist - pool_dist).clip(lower=0.0) / real_dist.replace(0.0, np.nan)).fillna(0.0)
        value_scores = (weight * surplus) - (weight * float(deficit_bias) * deficit)

        row_scores += pool_feature.map(value_scores).fillna(0.0).to_numpy(dtype=float)
        feature_count += 1
        if not feature_name.endswith("__missing"):
            protected_feature_series[feature_name] = pool_feature

    if feature_count == 0:
        return pool.head(target_rows).reset_index(drop=True)

    rng = np.random.default_rng(42 if random_seed is None else int(random_seed))
    row_scores += rng.normal(loc=0.0, scale=1e-9, size=len(pool))
    drop_order = np.argsort(-row_scores, kind="stable")

    keep_mask = np.ones(len(pool), dtype=bool)
    dropped = 0
    protected_counts = {
        feature_name: feature_series.value_counts(sort=False).to_dict()
        for feature_name, feature_series in protected_feature_series.items()
    }

    for row_index in drop_order:
        if dropped >= excess:
            break

        can_drop = True
        for feature_name, feature_series in protected_feature_series.items():
            value = feature_series.iat[int(row_index)]
            if protected_counts[feature_name].get(value, 0) <= 1:
                can_drop = False
                break

        if not can_drop:
            continue

        keep_mask[int(row_index)] = False
        dropped += 1
        for feature_name, feature_series in protected_feature_series.items():
            value = feature_series.iat[int(row_index)]
            protected_counts[feature_name][value] = protected_counts[feature_name].get(value, 0) - 1

    if dropped < excess:
        for row_index in drop_order:
            if dropped >= excess:
                break
            row_idx = int(row_index)
            if not keep_mask[row_idx]:
                continue
            keep_mask[row_idx] = False
            dropped += 1

    return pool.loc[keep_mask].reset_index(drop=True).head(target_rows)


def _iter_balancing_features(
    real_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    schema: Schema,
    max_unique: int,
    missingness_weight: float,
):
    for column in schema.columns:
        column_name = column.name
        if column_name not in candidate_df.columns:
            continue

        if column.mixed_value_kind is not None:
            _, _, real_states = split_mixed_column(
                column_name=column_name,
                series=real_df[column_name],
                source_kind=column.mixed_value_kind,
            )
            _, _, candidate_states = split_mixed_column(
                column_name=column_name,
                series=candidate_df[column_name],
                source_kind=column.mixed_value_kind,
            )
            yield (
                f"{column_name}__state",
                _normalized_feature_series(real_states),
                _normalized_feature_series(candidate_states),
                1.0,
            )
            continue

        if column.kind in {"categorical", "binary", "id_like"} and column.unique_count <= max_unique:
            yield (
                column_name,
                _normalized_feature_series(real_df[column_name]),
                _normalized_feature_series(candidate_df[column_name]),
                1.0,
            )
            continue

        if missingness_weight > 0.0 and 0.0 < float(column.missing_rate) < 1.0:
            yield (
                f"{column_name}__missing",
                _missingness_feature_series(real_df[column_name]),
                _missingness_feature_series(candidate_df[column_name]),
                float(missingness_weight),
            )


def _normalized_feature_series(series: pd.Series) -> pd.Series:
    return series.astype("object").where(series.notna(), "__MISSING__")


def _missingness_feature_series(series: pd.Series) -> pd.Series:
    values = np.where(series.isna(), "__MISSING__", "__PRESENT__")
    return pd.Series(values, index=series.index, dtype="object")


def _model_config_for_name(run_config: dict, model_name: str) -> dict:
    model_config = run_config.get(model_name, {})
    if isinstance(model_config, dict):
        return merge_dicts(run_config, model_config)
    return dict(run_config)


def _reseed_model(model: BaseSynthesizer, sample_seed: int) -> None:
    if hasattr(model, "seed"):
        model.seed = int(sample_seed)
    if hasattr(model, "rng"):
        model.rng = np.random.default_rng(sample_seed)

    for attribute_name in ("copula_model", "ctgan_model"):
        child = getattr(model, attribute_name, None)
        if child is not None:
            _reseed_model(child, sample_seed)


def _sha256_file(path: str | Path) -> str:
    file_path = Path(path)
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a submission CSV from a saved model artifact.")
    parser.add_argument("--model-path", required=True, help="Path to a saved model.pkl artifact.")
    parser.add_argument("--config", default="configs/base.yaml", help="Config path used to resolve repair/privacy settings.")
    parser.add_argument("--data", default="data/data.csv", help="Path to the source CSV.")
    parser.add_argument("--output", help="Output CSV path. Defaults to data/outputs/<model>_submission_<timestamp>.csv")
    parser.add_argument("--n-rows", type=int, help="Optional explicit row count.")
    parser.add_argument("--sample-seed", type=int, help="Optional reseed for sampling from a saved artifact.")
    args = parser.parse_args()

    result = run_generation(
        model_path=args.model_path,
        config_path=args.config,
        data_path=args.data,
        output_path=args.output,
        n_rows=args.n_rows,
        sample_seed=args.sample_seed,
    )
    print(result)

    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
