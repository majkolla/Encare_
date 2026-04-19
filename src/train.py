from __future__ import annotations

import argparse
from pathlib import Path

from src.data.loader import load_source_csv
from src.data.schema import infer_schema
from src.data.split import make_train_val_split
from src.eval.reports import write_run_report
from src.eval.score import compute_total_score
from src.generate import generate_synthetic_dataset
from src.rules.constraints import build_default_constraints
from src.utils.io import ensure_dir, merge_dicts, read_config, write_json
from src.utils.paths import resolve_repo_path
from src.utils.registry import create_model
from src.utils.seed import set_global_seed


def run_training(model_name: str, config_path: str, data_path: str) -> dict:
    base_config_path = resolve_repo_path("configs/base.yaml")
    config_path_obj = resolve_repo_path(config_path)
    data_path_obj = resolve_repo_path(data_path)

    base_config = read_config(base_config_path)
    run_config = merge_dicts(base_config, read_config(config_path_obj))
    seed = int(run_config.get("seed", 42))
    set_global_seed(seed)

    df = load_source_csv(data_path_obj)
    schema = infer_schema(df)
    constraints = build_default_constraints(df, schema)
    train_df, val_df = make_train_val_split(df, seed=seed, val_frac=run_config.get("val_frac", 0.2))

    model = create_model(model_name, seed=seed)
    model_config = _model_config_for_name(run_config, model_name)
    model.fit(train_df, schema, model_config)

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

    output_dir = ensure_dir(resolve_repo_path(run_config.get("artifact_dir", "data/artifacts")) / model_name)
    model_path = model.save(output_dir / "model.pkl")
    write_json(schema.to_dict(), output_dir / "schema.json")
    write_run_report(output_dir, model_name, metrics)

    return {
        "metrics": metrics,
        "model_path": str(model_path),
        "artifact_dir": str(output_dir),
        "note": "This command trains on the train split for validation and saves artifacts only. It does not write a submission CSV.",
        "next_step": f"python -m src.generate --model-path {model_path} --config {config_path_obj} --data {data_path_obj}",
    }


def _model_config_for_name(run_config: dict, model_name: str) -> dict:
    model_config = run_config.get(model_name, {})
    if isinstance(model_config, dict):
        return merge_dicts(run_config, model_config)
    return dict(run_config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one synthetic-data model on a train/validation split.")
    parser.add_argument("--model", required=True, help="Model name: baseline, copula, ctgan, hybrid.")
    parser.add_argument("--config", required=True, help="Path to a YAML or JSON config file.")
    parser.add_argument("--data", default="data/data.csv", help="Path to the source CSV.")
    args = parser.parse_args()

    result = run_training(args.model, args.config, args.data)
    print(result)


if __name__ == "__main__":
    main()
