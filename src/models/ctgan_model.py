from __future__ import annotations

from dataclasses import dataclass
import warnings
from typing import Any

import numpy as np
import pandas as pd

from src.models.base import BaseSynthesizer
from src.utils.types import Schema

_VALUE_STATE_TOKEN = "__VALUE__"
_MISSING_STATE_TOKEN = "__MISSING__"


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
    constant_columns: dict[str, _ConstantColumnState]
    datetime_columns: list[str]
    mixed_token_columns: dict[str, _MixedTokenColumnState]
    numeric_supports: dict[str, np.ndarray]
    time_columns: list[str]
    training_columns: list[str]


class CTGANSynthesizer(BaseSynthesizer):
    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.schema: Schema | None = None
        self.model = None
        self.config: dict[str, Any] = {}
        self.restore_state = _RestoreState(
            constant_columns={},
            datetime_columns=[],
            mixed_token_columns={},
            numeric_supports={},
            time_columns=[],
            training_columns=[],
        )

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        restore_state = getattr(self, "restore_state", None)
        constant_columns: dict[str, _ConstantColumnState] = {}
        mixed_token_columns: dict[str, _MixedTokenColumnState] = {}

        if restore_state is not None:
            for column_name, column_state in getattr(restore_state, "constant_columns", {}).items():
                if isinstance(column_state, _ConstantColumnState):
                    constant_columns[column_name] = column_state
                else:
                    constant_columns[column_name] = _ConstantColumnState(
                        value=column_state,
                        missing_rate=0.0,
                    )

            for column_name, column_state in getattr(restore_state, "mixed_token_columns", {}).items():
                if isinstance(column_state, _MixedTokenColumnState):
                    mixed_token_columns[column_name] = column_state
                    continue

                if isinstance(column_state, dict):
                    mixed_token_columns[column_name] = _MixedTokenColumnState(
                        source_kind=str(column_state.get("source_kind", "numeric")),
                        value_column=str(column_state.get("value_column", f"{column_name}__value")),
                        state_column=str(column_state.get("state_column", f"{column_name}__state")),
                        numeric_support=np.asarray(
                            column_state.get("numeric_support", []),
                            dtype=float,
                        ),
                        numeric_support_strings=list(
                            column_state.get("numeric_support_strings", []),
                        ),
                    )

        self.restore_state = _RestoreState(
            constant_columns=constant_columns,
            datetime_columns=list(getattr(restore_state, "datetime_columns", [])),
            mixed_token_columns=mixed_token_columns,
            numeric_supports=dict(getattr(restore_state, "numeric_supports", {})),
            time_columns=list(getattr(restore_state, "time_columns", [])),
            training_columns=list(getattr(restore_state, "training_columns", [])),
        )

    @staticmethod
    def is_backend_available() -> bool:
        try:
            from sdv.metadata import SingleTableMetadata  # noqa: F401
            from sdv.single_table import CTGANSynthesizer as _Backend  # noqa: F401
        except ImportError:
            return False
        return True

    def fit(self, df: pd.DataFrame, schema: Schema, config: dict[str, Any]) -> "CTGANSynthesizer":
        self.schema = schema
        self.config = config

        try:
            from sdv.metadata import SingleTableMetadata
            from sdv.single_table import CTGANSynthesizer as Backend
        except ImportError as exc:
            raise RuntimeError(
                "CTGAN backend not available. Install `sdv` to enable the CTGAN model."
            ) from exc

        training_df = self._prepare_training_dataframe(df, schema, config)

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=training_df)
        resolved_cuda = _resolve_cuda_setting(config.get("cuda", "auto"))
        batch_size = _resolve_batch_size(
            requested_batch_size=int(config.get("batch_size", 64)),
            pac=int(config.get("pac", 2)),
            n_rows=len(training_df),
        )
        backend_kwargs = {
            "metadata": metadata,
            "epochs": int(config.get("epochs", 100)),
            "batch_size": batch_size,
            "verbose": config.get("verbose", True),
            "embedding_dim": int(config.get("embedding_dim", 32)),
            "generator_dim": tuple(config.get("generator_dim", [64, 64])),
            "discriminator_dim": tuple(config.get("discriminator_dim", [64, 64])),
            "discriminator_steps": int(config.get("discriminator_steps", 1)),
            "log_frequency": bool(config.get("log_frequency", False)),
            "pac": int(config.get("pac", 1)),
            "enable_gpu": resolved_cuda,
            "enforce_rounding": config.get("enforce_rounding", True),
        }
        try:
            self.model = Backend(**backend_kwargs)
        except TypeError:
            backend_kwargs.pop("enable_gpu", None)
            backend_kwargs["cuda"] = resolved_cuda
            try:
                self.model = Backend(**backend_kwargs)
            except TypeError:
                backend_kwargs.pop("enforce_rounding", None)
                self.model = Backend(**backend_kwargs)
        self.model.fit(training_df)
        return self

    def sample(self, n_rows: int) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model must be fit before sampling.")
        sampled = self.model.sample(num_rows=n_rows)
        return self._restore_sampled_dataframe(sampled)

    def _prepare_training_dataframe(
        self,
        df: pd.DataFrame,
        schema: Schema,
        config: dict[str, Any],
    ) -> pd.DataFrame:
        processed: dict[str, pd.Series] = {}
        constant_columns: dict[str, _ConstantColumnState] = {}
        datetime_columns: list[str] = []
        mixed_token_columns: dict[str, _MixedTokenColumnState] = {}
        numeric_supports: dict[str, np.ndarray] = {}
        time_columns: list[str] = []

        for column in schema.columns:
            series = df[column.name]

            if column.kind == "constant" and config.get("drop_constant_columns", True):
                non_null = series.dropna()
                constant_columns[column.name] = _ConstantColumnState(
                    value=non_null.iloc[0] if not non_null.empty else np.nan,
                    missing_rate=float(series.isna().mean()),
                )
                continue

            mixed_state = _prepare_mixed_token_column(column.name, series, column.kind, config)
            if mixed_state is not None:
                column_state, value_series, state_series = mixed_state
                processed[column_state.value_column] = value_series
                processed[column_state.state_column] = state_series
                mixed_token_columns[column.name] = column_state
                continue

            if column.kind == "numeric":
                numeric_series = pd.to_numeric(series, errors="coerce")
                processed[column.name] = numeric_series
                support = _numeric_support_values(numeric_series, column.unique_count, config)
                if support.size > 0:
                    numeric_supports[column.name] = support
                continue

            if column.kind == "datetime":
                processed[column.name] = _datetime_to_numeric(series)
                datetime_columns.append(column.name)
                continue

            if column.kind == "time":
                processed[column.name] = _time_to_numeric(series)
                time_columns.append(column.name)
                continue

            processed[column.name] = series.astype("object")

        training_df = pd.DataFrame(processed, index=df.index)

        max_training_rows = config.get("max_training_rows")
        if max_training_rows:
            training_df = training_df.sample(
                n=min(int(max_training_rows), len(training_df)),
                random_state=self.seed,
            ).reset_index(drop=True)
        else:
            training_df = training_df.reset_index(drop=True)

        self.restore_state = _RestoreState(
            constant_columns=constant_columns,
            datetime_columns=datetime_columns,
            mixed_token_columns=mixed_token_columns,
            numeric_supports=numeric_supports,
            time_columns=time_columns,
            training_columns=list(training_df.columns),
        )
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
                restored_columns[column_name] = _restore_mixed_token_column(
                    sampled_df,
                    self.restore_state.mixed_token_columns[column_name],
                )
                continue

            if column_name not in sampled_df.columns:
                restored_columns[column_name] = np.nan
                continue

            if column_name in self.restore_state.datetime_columns:
                restored_columns[column_name] = _numeric_to_datetime(sampled_df[column_name])
                continue

            if column_name in self.restore_state.time_columns:
                restored_columns[column_name] = sampled_df[column_name].apply(_seconds_to_time)
                continue

            if column.kind == "numeric":
                restored_columns[column_name] = _snap_numeric_to_support(
                    sampled_df[column_name],
                    self.restore_state.numeric_supports.get(column_name),
                )
                continue

            restored_columns[column_name] = sampled_df[column_name]

        restored = pd.DataFrame(restored_columns, index=sampled_df.index)
        return restored[self.schema.column_order]


def _resolve_cuda_setting(value: Any) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "auto":
            available = _torch_cuda_available()
            if not available:
                warnings.warn(
                    "CTGAN `cuda: auto` resolved to CPU because torch.cuda.is_available() is False.",
                    stacklevel=2,
                )
            return available
        if normalized in {"true", "1", "yes", "on"}:
            value = True
        elif normalized in {"false", "0", "no", "off"}:
            value = False

    requested = bool(value)
    if requested and not _torch_cuda_available():
        warnings.warn(
            "CTGAN CUDA was requested, but torch.cuda.is_available() is False. Falling back to CPU.",
            stacklevel=2,
        )
        return False
    return requested


def _torch_cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _resolve_batch_size(requested_batch_size: int, pac: int, n_rows: int) -> int:
    safe_batch = max(2, min(requested_batch_size, n_rows))
    pac = max(1, pac)

    if safe_batch < pac:
        safe_batch = pac

    remainder = safe_batch % pac
    if remainder != 0:
        safe_batch -= remainder

    if safe_batch < pac:
        safe_batch = pac

    return max(2, safe_batch)


def _prepare_mixed_token_column(
    column_name: str,
    series: pd.Series,
    source_kind: str,
    config: dict[str, Any],
) -> tuple[_MixedTokenColumnState, pd.Series, pd.Series] | None:
    parsed_kind, value_series, token_mask = _detect_mixed_token_series(series, source_kind, config)
    if parsed_kind is None or value_series is None:
        return None

    state_series = pd.Series(_VALUE_STATE_TOKEN, index=series.index, dtype="object")
    state_series.loc[series.isna()] = _MISSING_STATE_TOKEN
    state_series.loc[token_mask] = (
        series.loc[token_mask]
        .astype("object")
        .astype(str)
        .str.strip()
    )

    numeric_support = np.array([], dtype=float)
    numeric_support_strings: list[str] = []
    if parsed_kind == "numeric":
        numeric_support, numeric_support_strings = _observed_numeric_string_support(
            series,
            value_series,
        )

    return (
        _MixedTokenColumnState(
            source_kind=parsed_kind,
            value_column=f"{column_name}__value",
            state_column=f"{column_name}__state",
            numeric_support=numeric_support,
            numeric_support_strings=numeric_support_strings,
        ),
        value_series,
        state_series,
    )


def _detect_mixed_token_series(
    series: pd.Series,
    source_kind: str,
    config: dict[str, Any],
) -> tuple[str | None, pd.Series | None, pd.Series]:
    if series.dropna().empty:
        return None, None, pd.Series(False, index=series.index)

    if source_kind == "datetime":
        numeric = _datetime_to_numeric(series)
        token_mask = series.notna() & numeric.isna()
        if token_mask.any() and numeric.notna().any():
            return "datetime", numeric, token_mask
        return None, None, token_mask

    if source_kind == "time":
        numeric = _time_to_numeric(series)
        token_mask = series.notna() & numeric.isna()
        if token_mask.any() and numeric.notna().any():
            return "time", numeric, token_mask
        return None, None, token_mask

    text = series.astype("object").where(series.notna(), None)
    numeric = pd.to_numeric(text, errors="coerce")
    token_mask = series.notna() & numeric.isna()
    if not token_mask.any() or not numeric.notna().any():
        return None, None, token_mask

    numeric_ratio = float(numeric.notna().mean())
    minimum_ratio = float(config.get("mixed_token_numeric_ratio_threshold", 0.20))
    if source_kind == "numeric" or numeric_ratio >= minimum_ratio:
        return "numeric", pd.Series(numeric, index=series.index, dtype="float64"), token_mask

    return None, None, token_mask


def _numeric_support_values(
    series: pd.Series,
    unique_count: int,
    config: dict[str, Any],
) -> np.ndarray:
    max_unique = int(config.get("snap_numeric_max_unique", 32))
    if unique_count > max_unique:
        return np.array([], dtype=float)

    observed = pd.to_numeric(series, errors="coerce").dropna().unique()
    if len(observed) == 0 or len(observed) > max_unique:
        return np.array([], dtype=float)

    return np.sort(np.asarray(observed, dtype=float))


def _observed_numeric_string_support(
    original_series: pd.Series,
    numeric_series: pd.Series,
) -> tuple[np.ndarray, list[str]]:
    numeric = pd.to_numeric(numeric_series, errors="coerce")
    valid_mask = original_series.notna() & numeric.notna()
    if not valid_mask.any():
        return np.array([], dtype=float), []

    observed = pd.DataFrame(
        {
            "numeric_value": numeric.loc[valid_mask].astype("float64"),
            "original_value": (
                original_series.loc[valid_mask]
                .astype("object")
                .astype(str)
                .str.strip()
            ),
        }
    )
    if observed.empty:
        return np.array([], dtype=float), []

    representatives = (
        observed.groupby(["numeric_value", "original_value"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(
            by=["numeric_value", "count", "original_value"],
            ascending=[True, False, True],
        )
        .drop_duplicates(subset=["numeric_value"], keep="first")
        .sort_values(by="numeric_value")
        .reset_index(drop=True)
    )
    return (
        representatives["numeric_value"].to_numpy(dtype=float),
        representatives["original_value"].tolist(),
    )


def _restore_mixed_token_column(
    sampled_df: pd.DataFrame,
    state: _MixedTokenColumnState,
) -> pd.Series:
    if state.state_column not in sampled_df.columns:
        return pd.Series(np.nan, index=sampled_df.index, dtype="object")

    token_series = sampled_df[state.state_column].astype("object")
    if state.value_column in sampled_df.columns:
        value_series = sampled_df[state.value_column]
    else:
        value_series = pd.Series(np.nan, index=sampled_df.index, dtype="float64")

    if state.source_kind == "numeric":
        restored_values = _restore_numeric_string_support(
            value_series,
            state.numeric_support,
            state.numeric_support_strings,
        )
    elif state.source_kind == "datetime":
        restored_values = _numeric_to_datetime(value_series).astype("object")
    elif state.source_kind == "time":
        restored_values = value_series.apply(_seconds_to_time).astype("object")
    else:
        restored_values = value_series.astype("object")

    restored = restored_values.where(token_series.eq(_VALUE_STATE_TOKEN), token_series)
    restored = restored.where(~token_series.eq(_MISSING_STATE_TOKEN), np.nan)
    return pd.Series(restored, index=sampled_df.index, dtype="object")


def _restore_numeric_string_support(
    series: pd.Series,
    support_values: np.ndarray,
    support_strings: list[str],
) -> pd.Series:
    snapped = _snap_numeric_to_support(series, support_values)
    if len(support_values) == 0 or len(support_values) != len(support_strings):
        return snapped.astype("object")

    lookup = {
        float(numeric_value): string_value
        for numeric_value, string_value in zip(support_values, support_strings)
    }
    restored_values: list[object] = []
    for value in snapped.to_numpy(dtype=float, na_value=np.nan):
        if np.isnan(value):
            restored_values.append(np.nan)
            continue
        restored_values.append(lookup.get(float(value), value))

    return pd.Series(restored_values, index=series.index, dtype="object")


def _sample_constant_column(
    state: _ConstantColumnState,
    n_rows: int,
    seed: int,
) -> pd.Series:
    values = pd.Series([state.value] * n_rows, dtype="object")
    if state.missing_rate <= 0.0:
        return values

    missing_count = int(round(state.missing_rate * n_rows))
    if missing_count <= 0:
        return values

    missing_count = min(missing_count, n_rows)
    rng = np.random.default_rng(seed)
    missing_positions = rng.choice(n_rows, size=missing_count, replace=False)
    values.iloc[missing_positions] = np.nan
    return values


def _snap_numeric_to_support(
    series: pd.Series,
    support_values: np.ndarray | None,
) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if support_values is None or len(support_values) == 0 or numeric.dropna().empty:
        return numeric

    snapped = numeric.to_numpy(dtype=float, na_value=np.nan)
    valid_mask = ~np.isnan(snapped)
    valid_values = snapped[valid_mask]
    insert_positions = np.searchsorted(support_values, valid_values, side="left")
    insert_positions = np.clip(insert_positions, 0, len(support_values) - 1)
    left_positions = np.clip(insert_positions - 1, 0, len(support_values) - 1)
    right_positions = insert_positions
    left_values = support_values[left_positions]
    right_values = support_values[right_positions]
    choose_left = np.abs(valid_values - left_values) <= np.abs(valid_values - right_values)
    snapped_values = np.where(choose_left, left_values, right_values)
    snapped[valid_mask] = snapped_values
    return pd.Series(snapped, index=series.index, dtype="float64")


def _stable_column_seed(column_name: str, base_seed: int) -> int:
    return int(base_seed + sum((idx + 1) * ord(char) for idx, char in enumerate(column_name)))


def _datetime_to_numeric(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    numeric = pd.Series(parsed.astype("int64"), index=series.index, dtype="float64")
    numeric[parsed.isna()] = np.nan
    return numeric / 1_000_000_000


def _numeric_to_datetime(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    parsed = pd.to_datetime(numeric, unit="s", errors="coerce")
    formatted = parsed.dt.strftime("%Y-%m-%d")
    return formatted.where(parsed.notna(), np.nan)


def _time_to_numeric(series: pd.Series) -> pd.Series:
    text = series.astype(str).where(series.notna(), None)
    split = text.str.split(":", expand=True)
    hours = pd.to_numeric(split[0], errors="coerce")
    minutes = pd.to_numeric(split[1], errors="coerce")
    return hours * 3600 + minutes * 60


def _seconds_to_time(value: float | int | None) -> str | float:
    if value is None or pd.isna(value):
        return np.nan

    total_seconds = int(round(float(value)))
    hours = max(min(total_seconds // 3600, 23), 0)
    minutes = max(min((total_seconds % 3600) // 60, 59), 0)
    return f"{hours:02d}:{minutes:02d}"
