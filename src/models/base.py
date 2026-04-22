from __future__ import annotations

import io
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class BaseSynthesizer(ABC):
    @abstractmethod
    def fit(self, df: pd.DataFrame, schema, config: dict[str, Any]) -> "BaseSynthesizer":
        raise NotImplementedError

    @abstractmethod
    def sample(self, n_rows: int) -> pd.DataFrame:
        raise NotImplementedError

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as handle:
            pickle.dump(self, handle)
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "BaseSynthesizer":
        input_path = Path(path)
        with input_path.open("rb") as handle:
            try:
                return pickle.load(handle)
            except RuntimeError as exc:
                if "Attempting to deserialize object on a CUDA device" not in str(exc):
                    raise

        with input_path.open("rb") as handle:
            return _TorchCPUUnpickler(handle).load()


class _TorchCPUUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "torch.storage" and name == "_load_from_bytes":
            try:
                import torch
            except ImportError:
                return super().find_class(module, name)

            return lambda b: torch.load(
                io.BytesIO(b),
                map_location="cpu",
                weights_only=False,
            )

        return super().find_class(module, name)
