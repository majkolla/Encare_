from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import warnings

import numpy as np
import pandas as pd

from src.data.mixed import (
    MixedColumnEncoding,
    datetime_to_numeric,
    numeric_to_datetime,
    numeric_support_values,
    restore_mixed_column,
    seconds_to_time,
    snap_numeric_to_support,
    split_mixed_column,
    time_to_numeric,
)
from src.models.base import BaseSynthesizer
from src.utils.types import Schema


@dataclass
class _ConstantColumnState:
    value: Any
    missing_rate: float


@dataclass
class _MixedTokenColumnState:
    source_kind: str
    value_column: str
    state_column: str
    numeric_support: np.ndarray
    numeric_support_strings: list[str]


@dataclass
class _RestoreState:
    constant_columns: dict[str, _ConstantColumnState] = field(default_factory=dict)
    datetime_columns: list[str] = field(default_factory=list)
    mixed_token_columns: dict[str, MixedColumnEncoding] = field(default_factory=dict)
    numeric_supports: dict[str, np.ndarray] = field(default_factory=dict)
    time_columns: list[str] = field(default_factory=list)
    training_columns: list[str] = field(default_factory=list)


class CTGANSynthesizer(BaseSynthesizer):
    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.schema: Schema | None = None
        self.model = None
        self.config: dict[str, Any] = {}
        self.restore_state = _RestoreState()

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.restore_state = _normalize_restore_state(getattr(self, "restore_state", None))

    @staticmethod
    def is_backend_available() -> bool:
        try:
            from sdv.metadata import SingleTableMetadata
            from sdv.single_table import CTGANSynthesizer as Backend
        except ImportError:
            return False
        return SingleTableMetadata is not None and Backend is not None

    def fit(self, df: pd.DataFrame, schema: Schema, config: dict[str, Any]) -> "CTGANSynthesizer":
        self.schema = schema
        self.config = config

        try:
            from sdv.metadata import SingleTableMetadata
            from sdv.single_table import CTGANSynthesizer as Backend
        except ImportError as exc:
            raise RuntimeError("Install `sdv` to use the CTGAN model.") from exc

        training_df = self._prepare_training_dataframe(df, schema, config)
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=training_df)
        self.model = _build_backend(Backend, metadata, config, len(training_df))
        self.model.fit(training_df)
        return self

    def sample(self, n_rows: int) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model must be fit before sampling.")
        return self._restore_sampled_dataframe(self.model.sample(num_rows=n_rows))

    def _prepare_training_dataframe(
        self,
        df: pd.DataFrame,
        schema: Schema,
        config: dict[str, Any],
    ) -> pd.DataFrame:
        processed: dict[str, pd.Series] = {}
        restore_state = _RestoreState()
        snap_max_unique = int(config.get("snap_numeric_max_unique", 32))

        for column in schema.columns:
            column_name = column.name
            series = df[column_name]

            if column.kind == "constant" and config.get("drop_constant_columns", True):
                restore_state.constant_columns[column_name] = _constant_column_state(series)
                continue

            mixed_kind = column.mixed_value_kind or _datetime_or_time_mixed_kind(series, column.kind)
            if mixed_kind is not None:
                encoding, value_series, state_series = split_mixed_column(
                    column_name=column_name,
                    series=series,
                    source_kind=mixed_kind,
                    snap_numeric_max_unique=snap_max_unique,
                )
                processed[encoding.value_column] = value_series
                processed[encoding.state_column] = state_series
                restore_state.mixed_token_columns[column_name] = encoding
                continue

            if column.kind == "numeric":
                numeric_series = pd.to_numeric(series, errors="coerce")
                processed[column_name] = numeric_series
                support = numeric_support_values(numeric_series, column.unique_count, snap_max_unique)
                if support.size:
                    restore_state.numeric_supports[column_name] = support
                continue

            if column.kind == "datetime":
                processed[column_name] = datetime_to_numeric(series)
                restore_state.datetime_columns.append(column_name)
                continue

            if column.kind == "time":
                processed[column_name] = time_to_numeric(series)
                restore_state.time_columns.append(column_name)
                continue

            processed[column_name] = series.astype("object")

        training_df = pd.DataFrame(processed, index=df.index)
        max_rows = config.get("max_training_rows")
        if max_rows:
            training_df = training_df.sample(
                n=min(int(max_rows), len(training_df)),
                random_state=self.seed,
            )
        training_df = training_df.reset_index(drop=True)

        restore_state.training_columns = list(training_df.columns)
        self.restore_state = restore_state
        return training_df

    def _restore_sampled_dataframe(self, sampled_df: pd.DataFrame) -> pd.DataFrame:
        if self.schema is None:
            raise RuntimeError("Schema is not available for post-processing.")

        restored_columns: dict[str, pd.Series | Any] = {}

        for column in self.schema.columns:
            column_name = column.name

            if column_name in self.restore_state.constant_columns:
                restored_columns[column_name] = _sample_constant_column(
                    self.restore_state.constant_columns[column_name],
                    len(sampled_df),
                    seed=_stable_column_seed(column_name, self.seed),
                )
                continue

            if column_name in self.restore_state.mixed_token_columns:
                restored_columns[column_name] = restore_mixed_column(
                    sampled_df,
                    self.restore_state.mixed_token_columns[column_name],
                )
                continue

            if column_name not in sampled_df.columns:
                restored_columns[column_name] = np.nan
                continue

            if column_name in self.restore_state.datetime_columns:
                restored_columns[column_name] = numeric_to_datetime(sampled_df[column_name])
                continue

            if column_name in self.restore_state.time_columns:
                restored_columns[column_name] = sampled_df[column_name].apply(seconds_to_time)
                continue

            if column.kind == "numeric":
                restored_columns[column_name] = snap_numeric_to_support(
                    sampled_df[column_name],
                    self.restore_state.numeric_supports.get(column_name),
                )
                continue

            restored_columns[column_name] = sampled_df[column_name]

        restored = pd.DataFrame(restored_columns, index=sampled_df.index)
        return restored[self.schema.column_order]


def _build_backend(backend_class, metadata, config: dict[str, Any], n_rows: int):
    kwargs = {
        "metadata": metadata,
        "epochs": int(config.get("epochs", 100)),
        "batch_size": _resolve_batch_size(
            requested_batch_size=int(config.get("batch_size", 64)),
            pac=int(config.get("pac", 1)),
            n_rows=n_rows,
        ),
        "verbose": config.get("verbose", True),
        "embedding_dim": int(config.get("embedding_dim", 32)),
        "generator_dim": tuple(config.get("generator_dim", [64, 64])),
        "discriminator_dim": tuple(config.get("discriminator_dim", [64, 64])),
        "discriminator_steps": int(config.get("discriminator_steps", 1)),
        "log_frequency": bool(config.get("log_frequency", False)),
        "pac": int(config.get("pac", 1)),
        "enable_gpu": _resolve_cuda_setting(config.get("cuda", "auto")),
        "enforce_rounding": config.get("enforce_rounding", True),
    }

    try:
        return backend_class(**kwargs)
    except TypeError:
        kwargs["cuda"] = kwargs.pop("enable_gpu")

    try:
        return backend_class(**kwargs)
    except TypeError:
        kwargs.pop("enforce_rounding", None)
        return backend_class(**kwargs)


def _constant_column_state(series: pd.Series) -> _ConstantColumnState:
    non_null = series.dropna()
    return _ConstantColumnState(
        value=non_null.iloc[0] if not non_null.empty else np.nan,
        missing_rate=float(series.isna().mean()),
    )


def _datetime_or_time_mixed_kind(series: pd.Series, source_kind: str) -> str | None:
    if source_kind == "datetime":
        numeric = datetime_to_numeric(series)
    elif source_kind == "time":
        numeric = time_to_numeric(series)
    else:
        return None

    has_token = bool((series.notna() & numeric.isna()).any())
    return source_kind if has_token and bool(numeric.notna().any()) else None


def _normalize_restore_state(restore_state: Any) -> _RestoreState:
    normalized = _RestoreState()
    if restore_state is None:
        return normalized

    for column_name, state in getattr(restore_state, "constant_columns", {}).items():
        if isinstance(state, _ConstantColumnState):
            normalized.constant_columns[column_name] = state
        else:
            normalized.constant_columns[column_name] = _ConstantColumnState(state, 0.0)

    for column_name, state in getattr(restore_state, "mixed_token_columns", {}).items():
        normalized.mixed_token_columns[column_name] = _normalize_mixed_state(column_name, state)

    normalized.datetime_columns = list(getattr(restore_state, "datetime_columns", []))
    normalized.numeric_supports = dict(getattr(restore_state, "numeric_supports", {}))
    normalized.time_columns = list(getattr(restore_state, "time_columns", []))
    normalized.training_columns = list(getattr(restore_state, "training_columns", []))
    return normalized


def _normalize_mixed_state(column_name: str, state: Any) -> MixedColumnEncoding:
    if isinstance(state, MixedColumnEncoding):
        return state

    payload = state if isinstance(state, dict) else state.__dict__
    return MixedColumnEncoding(
        name=str(payload.get("name", column_name)),
        source_kind=str(payload.get("source_kind", "numeric")),
        value_column=str(payload.get("value_column", f"{column_name}__value")),
        state_column=str(payload.get("state_column", f"{column_name}__state")),
        numeric_support=np.asarray(payload.get("numeric_support", []), dtype=float),
        numeric_support_strings=list(payload.get("numeric_support_strings", [])),
    )


def _resolve_cuda_setting(value: Any) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "auto":
            available = _torch_cuda_available()
            if not available:
                warnings.warn("CTGAN is using CPU because CUDA is not available.", stacklevel=2)
            return available
        if normalized in {"true", "1", "yes", "on"}:
            value = True
        elif normalized in {"false", "0", "no", "off"}:
            value = False

    requested = bool(value)
    if requested and not _torch_cuda_available():
        warnings.warn("CTGAN CUDA was requested but is not available. Using CPU.", stacklevel=2)
        return False
    return requested


def _torch_cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _resolve_batch_size(requested_batch_size: int, pac: int, n_rows: int) -> int:
    pac = max(1, pac)
    batch_size = max(2, min(max(requested_batch_size, pac), n_rows))
    batch_size -= batch_size % pac
    return max(2, pac, batch_size)


def _sample_constant_column(state: _ConstantColumnState, n_rows: int, seed: int) -> pd.Series:
    values = pd.Series([state.value] * n_rows, dtype="object")
    missing_count = min(int(round(state.missing_rate * n_rows)), n_rows)
    if missing_count <= 0:
        return values

    rng = np.random.default_rng(seed)
    values.iloc[rng.choice(n_rows, size=missing_count, replace=False)] = np.nan
    return values


def _stable_column_seed(column_name: str, base_seed: int) -> int:
    return int(base_seed + sum((idx + 1) * ord(char) for idx, char in enumerate(column_name)))
