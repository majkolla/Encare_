import argparse

import numpy as np
import pandas as pd
from scipy import stats

from data_validation import validate_submission
from src.data.schema import infer_schema
from src.eval.score import compute_total_score
from src.rules.constraints import build_default_constraints
from src.utils.io import merge_dicts, read_config
from src.utils.paths import resolve_repo_path


AGE_COL = "Age::40"
WEIGHT_COL = "Preoperative body weight (kg)::20"
HEIGHT_COL = "Height (cm)::23"
BMI_COL = "BMI::24"
GENDER_COL = "Gender::5"


def run_evaluation_report(original_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("EVALUATION REPORT: SYNTHETIC DATA FIDELITY")
    print("=" * 60)

    similarities = _compare_distributions(original_df, synthetic_df)
    print("\nStatistical Similarity Scores (1.0 = Identical):")
    valid_scores = {column: score for column, score in similarities.items() if not np.isnan(score)}
    for column, score in list(valid_scores.items())[:10]:
        print(f"  {column[:30]:30s}: {score:.3f}")

    validations = _validate_clinical_logic(synthetic_df)
    print("\nClinical Sanity Check Results:")
    for check, passed in validations.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {check:30s}: {status}")


def run_precheck(
    original_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    config_path: str = "configs/base.yaml",
    quick: bool = False,
) -> dict:
    print("\n" + "=" * 60)
    print("SUBMISSION PRECHECK")
    print("=" * 60)

    validation_errors = validate_submission(original_df, synthetic_df)
    if validation_errors:
        print("\nSubmission Format Check: FAILED")
        for error in validation_errors:
            print(f"  - {error}")
    else:
        print("\nSubmission Format Check: PASSED")

    run_evaluation_report(original_df, synthetic_df)

    if quick:
        print("\nAdvanced Local Score: SKIPPED")
        print("  Quick mode only runs submission-format and sanity checks.")
        return {"validation_errors": validation_errors, "metrics": None}

    if list(original_df.columns) != list(synthetic_df.columns):
        print("\nAdvanced Local Score: SKIPPED")
        print("  Column names/order must match the source CSV to compute the full local proxy score.")
        return {"validation_errors": validation_errors, "metrics": None}

    config = _load_score_config(config_path)
    schema = infer_schema(original_df)
    constraints = build_default_constraints(original_df, schema)
    metrics = compute_total_score(
        real_df=original_df,
        syn_df=synthetic_df,
        schema=schema,
        constraints=constraints,
        weights=config["score_weights"],
    )

    print("\nAdvanced Local Score:")
    print(f"  Total Score     : {metrics['total_score']:.4f}")
    print(f"  Marginal        : {metrics['marginal']['score']:.4f}")
    print(f"  Dependency      : {metrics['dependency']['score']:.4f}")
    print(f"  Discriminator   : {metrics['discriminator']['score']:.4f} (AUC={metrics['discriminator']['auc']:.4f})")
    print(f"  Privacy         : {metrics['privacy']['score']:.4f}")
    print(f"  Logic           : {metrics['logic']['score']:.4f}")

    print("\nNote:")
    print("  This is a local proxy only. It is useful for ranking runs, not for predicting the exact hackathon score.")
    return {"validation_errors": validation_errors, "metrics": metrics}


def _compare_distributions(orig: pd.DataFrame, synth: pd.DataFrame) -> dict[str, float]:
    results: dict[str, float] = {}
    common_cols = sorted(set(orig.columns) & set(synth.columns))

    for column in common_cols:
        original_numeric = pd.to_numeric(orig[column], errors="coerce").dropna()
        synthetic_numeric = pd.to_numeric(synth[column], errors="coerce").dropna()

        if len(original_numeric) == 0:
            results[column] = np.nan
            continue

        statistic, _ = stats.ks_2samp(original_numeric, synthetic_numeric)
        results[column] = round(1 - statistic, 3)

    return results


def _validate_clinical_logic(df: pd.DataFrame) -> dict[str, bool]:
    results: dict[str, bool] = {}

    if AGE_COL in df.columns:
        ages = pd.to_numeric(df[AGE_COL], errors="coerce").dropna()
        if len(ages) > 0:
            results["Age Within Range (18-110)"] = bool(((ages >= 18) & (ages <= 110)).all())

    if all(column in df.columns for column in [WEIGHT_COL, HEIGHT_COL, BMI_COL]):
        weight = pd.to_numeric(df[WEIGHT_COL], errors="coerce")
        height_meters = pd.to_numeric(df[HEIGHT_COL], errors="coerce") / 100
        bmi = pd.to_numeric(df[BMI_COL], errors="coerce")
        valid_mask = (height_meters > 0) & height_meters.notna() & weight.notna() & bmi.notna()

        if valid_mask.any():
            expected_bmi = weight[valid_mask] / (height_meters[valid_mask] ** 2)
            actual_bmi = bmi[valid_mask]
            diff = abs(actual_bmi - expected_bmi)
            results["BMI-Weight-Height Logic"] = bool((diff < 2.0).mean() > 0.90)

    if GENDER_COL in df.columns:
        allowed_values = {"Male", "Female", "Unknown"}
        results["Valid Gender Categories"] = bool(df[GENDER_COL].isin(allowed_values).all())

    return results


def _load_score_config(config_path: str) -> dict:
    default_config = {"score_weights": {
        "marginal": 0.30,
        "dependency": 0.30,
        "discriminator": 0.20,
        "privacy": 0.15,
        "logic": 0.05,
    }}

    config_file = resolve_repo_path(config_path)
    if not config_file.exists():
        return default_config

    return merge_dicts(default_config, read_config(config_file))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local precheck on a generated synthetic CSV.")
    parser.add_argument("--original", default="data/data.csv", help="Path to the original/source CSV.")
    parser.add_argument("--synthetic", required=True, help="Path to the generated synthetic CSV.")
    parser.add_argument("--config", default="configs/base.yaml", help="Config path for score weights.")
    parser.add_argument("--quick", action="store_true", help="Skip the slower full local score and only run fast checks.")
    args = parser.parse_args()

    original_path = resolve_repo_path(args.original)
    synthetic_path = resolve_repo_path(args.synthetic)

    original_df = pd.read_csv(original_path, low_memory=False)
    synthetic_df = pd.read_csv(synthetic_path, low_memory=False)
    run_precheck(original_df, synthetic_df, config_path=args.config, quick=args.quick)


if __name__ == "__main__":
    main()
