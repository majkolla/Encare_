from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from src.data.loader import load_source_csv
from src.data.schema import infer_schema
from src.data.split import make_train_val_split
from src.eval.reports import write_comparison_report, write_run_report
from src.eval.score import compare_runs, compute_total_score
from src.generate import generate_and_save_submission, generate_synthetic_dataset
from src.rules.constraints import build_default_constraints
from src.utils.io import ensure_dir, merge_dicts, read_config, write_json
from src.utils.logging import get_logger
from src.utils.paths import resolve_repo_path
from src.utils.registry import create_model
from src.utils.seed import set_global_seed
from src.utils.types import RunResult


def run_pipeline(config_path: str, data_path: str) -> dict:
    logger = get_logger("encare.pipeline")
    base_config_path = resolve_repo_path("configs/base.yaml")
    config_path_obj = resolve_repo_path(config_path)
    data_path_obj = resolve_repo_path(data_path)

    base_config = read_config(base_config_path)
    run_config = merge_dicts(base_config, read_config(config_path_obj))
    seed = int(run_config.get("seed", 42))
    set_global_seed(seed)

    data = load_source_csv(data_path_obj)
    schema = infer_schema(data)
    constraints = build_default_constraints(data, schema)
    train_df, val_df = make_train_val_split(data, seed=seed, val_frac=run_config.get("val_frac", 0.2))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(resolve_repo_path(run_config.get("output_dir", "data/outputs")) / f"run_{timestamp}")
    artifact_dir = ensure_dir(resolve_repo_path(run_config.get("artifact_dir", "data/artifacts")) / f"run_{timestamp}")

    write_json(schema.to_dict(), artifact_dir / "schema.json")
    logger.info("Loaded data with %s rows and %s columns.", len(data), len(data.columns))

    results: list[RunResult] = []

    for model_name in run_config.get("models", ["baseline", "copula", "ctgan", "hybrid"]):
        logger.info("Training %s", model_name)
        try:
            model = create_model(model_name, seed=seed)
            model_config = _model_config_for_name(run_config, model_name)
            model.fit(train_df, schema, model_config)

            if model_name == "hybrid" and hasattr(model, "grid_search_alpha"):
                best_alpha, alpha_results = model.grid_search_alpha(
                    train_df=train_df,
                    val_df=val_df,
                    schema=schema,
                    constraints=constraints,
                    weights=run_config["score_weights"],
                    alphas=model_config.get("alphas"),
                )
                logger.info("Hybrid best alpha: %.2f", best_alpha)
                write_json({"alpha_search": alpha_results, "best_alpha": best_alpha}, artifact_dir / "hybrid_alpha_search.json")

            synthetic_val = generate_synthetic_dataset(
                model=model,
                real_df=train_df,
                schema=schema,
                constraints=constraints,
                n_rows=len(val_df),
                repair=model_config.get("repair", False),
                privacy_filter=model_config.get("privacy_filter", False),
                privacy_min_distance=model_config.get("privacy_min_distance", 0.0),
            )
            metrics = compute_total_score(val_df, synthetic_val, schema, constraints, run_config["score_weights"])
            model_artifact_dir = ensure_dir(artifact_dir / model_name)
            model_path = model.save(model_artifact_dir / "model.pkl")
            write_run_report(model_artifact_dir, model_name, metrics)
            results.append(
                RunResult(
                    model_name=model_name,
                    metrics=metrics,
                    artifact_path=str(model_path),
                )
            )
        except Exception as exc:
            logger.warning("Skipping %s: %s", model_name, exc)
            results.append(
                RunResult(
                    model_name=model_name,
                    metrics={"total_score": float("-inf")},
                    notes=[str(exc)],
                )
            )

    ranked = compare_runs([result for result in results if result.total_score != float("-inf")])
    write_comparison_report(run_dir, ranked)

    successful = [result for result in results if result.total_score != float("-inf")]
    if not successful:
        raise RuntimeError("No models completed successfully.")

    best_result = max(successful, key=lambda result: result.total_score)
    logger.info("Best model: %s", best_result.model_name)

    best_model = create_model(best_result.model_name, seed=seed)
    best_config = _model_config_for_name(run_config, best_result.model_name)
    best_model.fit(data, schema, best_config)
    submission_path = run_dir / f"{best_result.model_name}_submission.csv"
    _, submission_errors = generate_and_save_submission(
        model=best_model,
        real_df=data,
        schema=schema,
        constraints=constraints,
        output_path=submission_path,
        n_rows=max(len(data), int(len(data) * run_config.get("n_rows_multiplier", 1.0))),
        repair=best_config.get("repair", False),
        privacy_filter=best_config.get("privacy_filter", False),
        privacy_min_distance=best_config.get("privacy_min_distance", 0.0),
    )

    summary = {
        "run_dir": str(run_dir),
        "artifact_dir": str(artifact_dir),
        "ranked_runs": ranked,
        "best_model": best_result.model_name,
        "submission_path": str(submission_path),
        "submission_errors": submission_errors,
    }
    write_json(summary, run_dir / "summary.json")
    return summary


def _model_config_for_name(run_config: dict, model_name: str) -> dict:
    model_config = run_config.get(model_name, {})
    if isinstance(model_config, dict):
        return merge_dicts(run_config, model_config)
    return dict(run_config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Encare synthetic data experiment pipeline.")
    parser.add_argument("--config", default="configs/base.yaml", help="Base experiment config.")
    parser.add_argument("--data", default="data/data.csv", help="Source CSV path.")
    args = parser.parse_args()

    summary = run_pipeline(args.config, args.data)
    print(summary)


if __name__ == "__main__":
    main()
