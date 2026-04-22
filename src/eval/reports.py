from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.io import ensure_dir, write_json, write_markdown


def write_run_report(
    output_dir: str | Path,
    model_name: str,
    metrics: dict[str, Any],
    diagnostics: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    output_path = ensure_dir(output_dir)
    json_path = write_json(metrics, output_path / f"{model_name}_metrics.json")
    if diagnostics is not None:
        write_json(diagnostics, output_path / f"{model_name}_diagnostics.json")
    markdown_path = write_markdown(
        _render_metrics_markdown(model_name, metrics, diagnostics),
        output_path / f"{model_name}_report.md",
    )
    return json_path, markdown_path


def write_comparison_report(
    output_dir: str | Path,
    ranked_runs: list[dict[str, Any]],
) -> tuple[Path, Path]:
    output_path = ensure_dir(output_dir)
    payload = {"ranked_runs": ranked_runs}
    json_path = write_json(payload, output_path / "comparison.json")
    markdown_path = write_markdown(_render_comparison_markdown(ranked_runs), output_path / "comparison.md")
    return json_path, markdown_path


def _render_metrics_markdown(
    model_name: str,
    metrics: dict[str, Any],
    diagnostics: dict[str, Any] | None = None,
) -> str:
    lines = [f"# {model_name}", ""]
    lines.append(f"- Total score: {metrics.get('total_score', 0.0):.4f}")
    for key in ["marginal", "dependency", "discriminator", "privacy", "logic"]:
        section = metrics.get(key, {})
        score = section.get("score")
        if score is not None:
            lines.append(f"- {key.title()} score: {score:.4f}")

    if diagnostics:
        lines.extend(["", "## Hard Slices", ""])
        for row in diagnostics.get("worst_numeric_columns", [])[:5]:
            lines.append(f"- Numeric: {row['column']} -> {row['score']:.4f}")
        for row in diagnostics.get("worst_categorical_columns", [])[:5]:
            lines.append(f"- Categorical: {row['column']} -> {row['score']:.4f}")

        lines.extend(["", "## Dependency Drift", ""])
        for row in diagnostics.get("worst_dependency_pairs", [])[:5]:
            lines.append(
                f"- {row['metric']}: {row['left']} vs {row['right']} -> |diff|={row['abs_diff']:.4f}"
            )

        lines.extend(["", "## Discriminator Drivers", ""])
        for row in diagnostics.get("discriminator", {}).get("top_features", [])[:5]:
            lines.append(f"- {row['feature']} -> {row['importance']:.4f}")
    return "\n".join(lines) + "\n"


def _render_comparison_markdown(ranked_runs: list[dict[str, Any]]) -> str:
    lines = ["# Model Comparison", ""]
    for index, run in enumerate(ranked_runs, start=1):
        lines.append(f"{index}. {run['model_name']}: {run['total_score']:.4f}")
    return "\n".join(lines) + "\n"
