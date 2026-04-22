from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io import ensure_dir


def load_source_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Source CSV not found: {csv_path}")
    return pd.read_csv(csv_path, low_memory=False)


def save_submission_csv(df: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    df.to_csv(output_path, index=False)
    return output_path

