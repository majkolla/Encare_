from __future__ import annotations

import argparse

from src.data.loader import load_source_csv
from src.data.schema import infer_schema
from src.utils.config import model_config_for_name
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

    model = create_model(model_name, seed=seed)
    model_config = model_config_for_name(run_config, model_name)
    model.fit(df, schema, model_config)

    output_dir = ensure_dir(resolve_repo_path(run_config.get("artifact_dir", "data/artifacts")) / model_name)
    model_path = model.save(output_dir / "model.pkl")
    write_json(schema.to_dict(), output_dir / "schema.json")
    training_summary = {
        "model_name": model_name,
        "model_path": str(model_path),
        "artifact_dir": str(output_dir),
        "config_path": str(config_path_obj),
        "data_path": str(data_path_obj),
        "rows_trained": len(df),
        "columns_trained": len(df.columns),
    }
    write_json(training_summary, output_dir / "training_summary.json")

    return {
        "model_name": model_name,
        "model_path": str(model_path),
        "artifact_dir": str(output_dir),
        "rows_trained": len(df),
        "note": "This command fits the selected model on the full source dataset and saves artifacts only. It does not compute internal scores or write a submission CSV.",
        "next_step": f"python -m src.generate --model-path {model_path} --config {config_path_obj} --data {data_path_obj}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one synthetic-data model on the full source dataset.")
    parser.add_argument("--model", required=True, help="Model name: baseline or copula.")
    parser.add_argument("--config", required=True, help="Path to a YAML or JSON config file.")
    parser.add_argument("--data", default="data/data.csv", help="Path to the source CSV.")
    args = parser.parse_args()

    result = run_training(args.model, args.config, args.data)
    print(result)


if __name__ == "__main__":
    main()
