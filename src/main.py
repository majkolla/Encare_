from __future__ import annotations

import argparse
from datetime import datetime

from src.data.loader import load_source_csv
from src.data.schema import infer_schema
from src.generate import generate_and_save_submission, write_output_notes
from src.rules.constraints import build_default_constraints
from src.utils.config import model_config_for_name
from src.utils.io import ensure_dir, merge_dicts, read_config, write_json
from src.utils.logging import get_logger
from src.utils.paths import resolve_repo_path
from src.utils.registry import create_model
from src.utils.seed import set_global_seed


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
    constraints = build_default_constraints(
        data,
        schema,
        include_conditional_blanks=bool(run_config.get("include_conditional_blanks", False)),
        derived_repair_mode=str(run_config.get("derived_repair_mode", "overwrite")),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(resolve_repo_path(run_config.get("output_dir", "data/outputs")) / f"run_{timestamp}")
    artifact_dir = ensure_dir(resolve_repo_path(run_config.get("artifact_dir", "data/artifacts")) / f"run_{timestamp}")

    write_json(schema.to_dict(), artifact_dir / "schema.json")
    logger.info("Loaded data with %s rows and %s columns.", len(data), len(data.columns))

    model_outputs: list[dict[str, object]] = []

    for model_name in _resolve_model_names(run_config):
        logger.info("Training %s", model_name)
        model = create_model(model_name, seed=seed)
        model_config = model_config_for_name(run_config, model_name)
        model.fit(data, schema, model_config)

        model_artifact_dir = ensure_dir(artifact_dir / model_name)
        model_path = model.save(model_artifact_dir / "model.pkl")
        write_json(schema.to_dict(), model_artifact_dir / "schema.json")

        submission_path = run_dir / f"{model_name}_submission.csv"
        _, submission_errors = generate_and_save_submission(
            model=model,
            real_df=data,
            schema=schema,
            constraints=constraints,
            output_path=submission_path,
            n_rows=max(len(data), int(len(data) * run_config.get("n_rows_multiplier", 1.0))),
            repair=model_config.get("repair", False),
            privacy_filter=model_config.get("privacy_filter", False),
            privacy_min_distance=model_config.get("privacy_min_distance", 0.0),
            privacy_min_distance_quantile=model_config.get("privacy_min_distance_quantile"),
        )
        submission_note_path = write_output_notes(
            output_path=submission_path,
            metadata={
                "model_name": model_name,
                "model_path": str(model_path),
                "config_path": str(config_path_obj),
                "data_path": str(data_path_obj),
                "n_rows": max(len(data), int(len(data) * run_config.get("n_rows_multiplier", 1.0))),
                "repair": model_config.get("repair", False),
                "privacy_filter": model_config.get("privacy_filter", False),
                "privacy_min_distance": model_config.get("privacy_min_distance", 0.0),
                "privacy_min_distance_quantile": model_config.get("privacy_min_distance_quantile"),
                "mixed_column_strategy": model_config.get("mixed_column_strategy"),
                "notes": [
                    "Generated via `python -m src.main` from a model trained on the full source dataset.",
                    "This pipeline does not run internal validation scoring or model ranking.",
                ],
            },
        )
        model_outputs.append(
            {
                "model_name": model_name,
                "model_path": str(model_path),
                "artifact_dir": str(model_artifact_dir),
                "submission_path": str(submission_path),
                "submission_note_path": str(submission_note_path),
                "submission_errors": submission_errors,
            }
        )

    if not model_outputs:
        raise RuntimeError("No models were configured for training.")

    summary = {
        "run_dir": str(run_dir),
        "artifact_dir": str(artifact_dir),
        "models": model_outputs,
        "primary_model": model_outputs[0]["model_name"],
        "primary_submission_path": model_outputs[0]["submission_path"],
    }
    write_json(summary, run_dir / "summary.json")
    return summary


def _resolve_model_names(run_config: dict) -> list[str]:
    configured_models = run_config.get("models")
    if isinstance(configured_models, list):
        model_names = [str(model_name) for model_name in configured_models if str(model_name).strip()]
        if model_names:
            return model_names

    configured_model = str(run_config.get("model", "copula")).strip()
    if configured_model:
        return [configured_model]

    return ["copula"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Encare synthetic data experiment pipeline.")
    parser.add_argument("--config", default="configs/base.yaml", help="Base experiment config.")
    parser.add_argument("--data", default="data/data.csv", help="Source CSV path.")
    args = parser.parse_args()

    summary = run_pipeline(args.config, args.data)
    print(summary)


if __name__ == "__main__":
    main()
