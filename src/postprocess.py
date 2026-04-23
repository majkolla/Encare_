from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.loader import load_source_csv
from src.data.schema import infer_schema
from src.rules.constraints import build_default_constraints
from src.rules.repair import softly_align_conditional_blank_rates
from src.submit import save_validated_submission
from src.utils.io import ensure_dir, write_markdown
from src.utils.paths import resolve_repo_path


def run_soft_conditional_postprocess(
    synthetic_path: str,
    data_path: str,
    output_path: str,
    strength: float,
    min_excess_rate: float,
    derived_repair_mode: str = "fill_missing_datetime",
    include_parent_patterns: list[str] | None = None,
    exclude_parent_patterns: list[str] | None = None,
    include_child_patterns: list[str] | None = None,
    exclude_child_patterns: list[str] | None = None,
    max_blanks_per_rule: int | None = None,
) -> dict[str, object]:
    synthetic_path_obj = resolve_repo_path(synthetic_path)
    data_path_obj = resolve_repo_path(data_path)
    output_path_obj = resolve_repo_path(output_path)

    real_df = load_source_csv(data_path_obj)
    schema = infer_schema(real_df)
    synthetic_df = pd.read_csv(synthetic_path_obj, low_memory=False)
    constraints = build_default_constraints(
        real_df,
        schema,
        include_conditional_blanks=True,
        derived_repair_mode=derived_repair_mode,
    )

    adjusted_df, summary_rows = softly_align_conditional_blank_rates(
        syn_df=synthetic_df,
        real_df=real_df,
        constraints=constraints,
        strength=strength,
        min_excess_rate=min_excess_rate,
        include_parents=include_parent_patterns,
        exclude_parents=exclude_parent_patterns,
        include_children=include_child_patterns,
        exclude_children=exclude_child_patterns,
        max_blanks_per_rule=max_blanks_per_rule,
    )

    saved_path, errors = save_validated_submission(
        adjusted_df,
        real_df,
        schema,
        output_path_obj,
    )
    note_path = _write_postprocess_notes(
        input_path=synthetic_path_obj,
        output_path=saved_path,
        strength=strength,
        min_excess_rate=min_excess_rate,
        source_rows=len(summary_rows),
        cell_changes=int((synthetic_df.fillna("__NA__") != adjusted_df.fillna("__NA__")).to_numpy().sum()),
        row_changes=int((synthetic_df.fillna("__NA__") != adjusted_df.fillna("__NA__")).any(axis=1).sum()),
        summary_rows=summary_rows,
        include_parent_patterns=include_parent_patterns,
        exclude_parent_patterns=exclude_parent_patterns,
        include_child_patterns=include_child_patterns,
        exclude_child_patterns=exclude_child_patterns,
        max_blanks_per_rule=max_blanks_per_rule,
    )
    return {
        "input_path": str(synthetic_path_obj),
        "output_path": str(saved_path),
        "note_path": str(note_path),
        "errors": errors,
        "rule_changes": len(summary_rows),
    }


def _write_postprocess_notes(
    input_path: Path,
    output_path: Path,
    strength: float,
    min_excess_rate: float,
    source_rows: int,
    cell_changes: int,
    row_changes: int,
    summary_rows: list[dict[str, object]],
    include_parent_patterns: list[str] | None = None,
    exclude_parent_patterns: list[str] | None = None,
    include_child_patterns: list[str] | None = None,
    exclude_child_patterns: list[str] | None = None,
    max_blanks_per_rule: int | None = None,
) -> Path:
    note_path = output_path.with_suffix(".md")
    lines = [
        f"# {output_path.name}",
        "",
        f"- Source CSV: {input_path}",
        f"- Postprocess: soft conditional blank alignment",
        f"- Strength: {strength}",
        f"- Minimum excess rate: {min_excess_rate}",
        f"- Changed rows: {row_changes}",
        f"- Changed cells: {cell_changes}",
        f"- Rule-child updates: {source_rows}",
    ]
    if include_parent_patterns:
        lines.append(f"- Included parent patterns: {', '.join(include_parent_patterns)}")
    if exclude_parent_patterns:
        lines.append(f"- Excluded parent patterns: {', '.join(exclude_parent_patterns)}")
    if include_child_patterns:
        lines.append(f"- Included child patterns: {', '.join(include_child_patterns)}")
    if exclude_child_patterns:
        lines.append(f"- Excluded child patterns: {', '.join(exclude_child_patterns)}")
    if max_blanks_per_rule is not None:
        lines.append(f"- Maximum blanks per rule: {max_blanks_per_rule}")
    if summary_rows:
        lines.extend(["", "## Top Adjustments", ""])
        ranked = sorted(summary_rows, key=lambda row: int(row["blanked_rows"]), reverse=True)
        for row in ranked[:15]:
            lines.append(
                f"- parent={row['parent']}, child={row['child']}, "
                f"syn_rate_before={float(row['syn_rate_before']):.4f}, "
                f"real_rate={float(row['real_rate']):.4f}, "
                f"target_rate={float(row['target_rate']):.4f}, "
                f"blanked_rows={int(row['blanked_rows'])}"
            )
    return write_markdown("\n".join(lines) + "\n", note_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Postprocess a synthetic CSV with lightweight structural adjustments.")
    parser.add_argument("--synthetic", required=True, help="Input synthetic CSV to adjust.")
    parser.add_argument("--data", default="data/data.csv", help="Path to the source CSV.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--strength", type=float, default=0.05, help="Fraction of the excess conditional-gap to remove.")
    parser.add_argument(
        "--min-excess-rate",
        type=float,
        default=0.05,
        help="Only adjust child fields whose inactive-parent fill rate exceeds the real rate by this amount.",
    )
    parser.add_argument(
        "--derived-repair-mode",
        default="fill_missing_datetime",
        help="Constraint mode used when building rule metadata.",
    )
    parser.add_argument(
        "--include-parent-pattern",
        action="append",
        default=[],
        help="Case-insensitive substring filter for parent rules to include. Repeatable.",
    )
    parser.add_argument(
        "--exclude-parent-pattern",
        action="append",
        default=[],
        help="Case-insensitive substring filter for parent rules to exclude. Repeatable.",
    )
    parser.add_argument(
        "--include-child-pattern",
        action="append",
        default=[],
        help="Case-insensitive substring filter for child columns to include. Repeatable.",
    )
    parser.add_argument(
        "--exclude-child-pattern",
        action="append",
        default=[],
        help="Case-insensitive substring filter for child columns to exclude. Repeatable.",
    )
    parser.add_argument(
        "--max-blanks-per-rule",
        type=int,
        default=None,
        help="Optional absolute cap on how many rows a single parent-child rule may blank.",
    )
    args = parser.parse_args()

    ensure_dir(resolve_repo_path(Path(args.output)).parent)
    result = run_soft_conditional_postprocess(
        synthetic_path=args.synthetic,
        data_path=args.data,
        output_path=args.output,
        strength=args.strength,
        min_excess_rate=args.min_excess_rate,
        derived_repair_mode=args.derived_repair_mode,
        include_parent_patterns=args.include_parent_pattern,
        exclude_parent_patterns=args.exclude_parent_pattern,
        include_child_patterns=args.include_child_pattern,
        exclude_child_patterns=args.exclude_child_pattern,
        max_blanks_per_rule=args.max_blanks_per_rule,
    )
    print(result)


if __name__ == "__main__":
    main()
