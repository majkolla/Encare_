from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.mixed import (
    MISSING_STATE_TOKEN,
    VALUE_STATE_TOKEN,
    datetime_to_numeric,
    numeric_to_datetime,
    seconds_to_time,
    split_mixed_column,
    time_to_numeric,
)
from src.utils.types import Preprocessor, Schema


def fit_preprocessor(df: pd.DataFrame, schema: Schema) -> Preprocessor:
    categorical_levels: dict[str, list[Any]] = {}
    numeric_fill_values: dict[str, float] = {}
    datetime_fill_values: dict[str, float] = {}
    time_fill_values: dict[str, float] = {}
    mixed_state_levels: dict[str, list[Any]] = {}
    mixed_value_fill_values: dict[str, float] = {}
    mixed_value_kinds: dict[str, str] = {}

    for column in schema.columns:
        series = df[column.name]

        if column.mixed_value_kind is not None:
            _, value_series, state_series = split_mixed_column(
                column_name=column.name,
                series=series,
                source_kind=column.mixed_value_kind,
            )
            mixed_state_levels[column.name] = list(
                state_series.dropna().astype("object").drop_duplicates().tolist()
            )
            mixed_value_fill_values[column.name] = _safe_fill_value(value_series)
            mixed_value_kinds[column.name] = column.mixed_value_kind
            continue

        if column.kind in {"categorical", "binary", "id_like"}:
            categorical_levels[column.name] = list(
                series.dropna().astype("object").drop_duplicates().tolist()
            )
        elif column.kind == "numeric":
            numeric_series = pd.to_numeric(series, errors="coerce")
            numeric_fill_values[column.name] = _safe_fill_value(numeric_series)
        elif column.kind == "datetime":
            numeric_series = datetime_to_numeric(series)
            datetime_fill_values[column.name] = _safe_fill_value(numeric_series)
        elif column.kind == "time":
            numeric_series = time_to_numeric(series)
            time_fill_values[column.name] = _safe_fill_value(numeric_series)

    return Preprocessor(
        schema=schema,
        categorical_levels=categorical_levels,
        numeric_fill_values=numeric_fill_values,
        datetime_fill_values=datetime_fill_values,
        time_fill_values=time_fill_values,
        mixed_state_levels=mixed_state_levels,
        mixed_value_fill_values=mixed_value_fill_values,
        mixed_value_kinds=mixed_value_kinds,
    )


def transform_for_model(
    df: pd.DataFrame,
    preprocessor: Preprocessor,
    model_name: str,
) -> pd.DataFrame:
    transformed_columns: dict[str, pd.Series] = {}
    numeric_output_columns: list[str] = []
    categorical_output_columns: list[str] = []

    for column in preprocessor.schema.columns:
        series = df[column.name]

        if column.mixed_value_kind is not None:
            mixed_kind = preprocessor.mixed_value_kinds.get(column.name, column.mixed_value_kind)
            encoding, value_series, state_series = split_mixed_column(
                column_name=column.name,
                series=series,
                source_kind=mixed_kind,
            )
            fill_value = preprocessor.mixed_value_fill_values.get(column.name, 0.0)
            transformed_columns[encoding.value_column] = value_series.fillna(fill_value)
            state_levels = preprocessor.mixed_state_levels.get(column.name, [MISSING_STATE_TOKEN])
            encoded_state = pd.Categorical(state_series.astype("object"), categories=state_levels)
            transformed_columns[encoding.state_column] = pd.Series(
                encoded_state.codes.astype(float),
                index=df.index,
            )
            numeric_output_columns.append(encoding.value_column)
            categorical_output_columns.append(encoding.state_column)
            continue

        if column.kind == "numeric":
            numeric_series = pd.to_numeric(series, errors="coerce")
            fill_value = preprocessor.numeric_fill_values.get(column.name, 0.0)
            transformed_columns[column.name] = numeric_series.fillna(fill_value)
            numeric_output_columns.append(column.name)
            continue

        if column.kind == "datetime":
            numeric_series = datetime_to_numeric(series)
            fill_value = preprocessor.datetime_fill_values.get(column.name, 0.0)
            transformed_columns[column.name] = numeric_series.fillna(fill_value)
            numeric_output_columns.append(column.name)
            continue

        if column.kind == "time":
            numeric_series = time_to_numeric(series)
            fill_value = preprocessor.time_fill_values.get(column.name, 0.0)
            transformed_columns[column.name] = numeric_series.fillna(fill_value)
            numeric_output_columns.append(column.name)
            continue

        levels = preprocessor.categorical_levels.get(column.name, [])
        encoded = pd.Categorical(series.astype("object"), categories=levels)
        encoded_series = pd.Series(encoded.codes.astype(float), index=df.index)
        encoded_series.loc[series.isna()] = np.nan
        transformed_columns[column.name] = encoded_series
        categorical_output_columns.append(column.name)

    transformed = pd.DataFrame(transformed_columns, index=df.index)

    if model_name in {"discriminator", "privacy"}:
        numeric_part = transformed[numeric_output_columns].astype(float) if numeric_output_columns else pd.DataFrame(index=df.index)
        categorical_part = (
            pd.get_dummies(
                transformed[categorical_output_columns]
                .apply(lambda series: series.astype("string"))
                .fillna(MISSING_STATE_TOKEN),
                dummy_na=False,
                drop_first=False,
            ).astype(float)
            if categorical_output_columns
            else pd.DataFrame(index=df.index)
        )
        return pd.concat([numeric_part, categorical_part], axis=1)

    return transformed


def inverse_transform(df_model: pd.DataFrame, preprocessor: Preprocessor) -> pd.DataFrame:
    restored_columns: dict[str, pd.Series | list[Any]] = {}

    for column in preprocessor.schema.columns:
        if column.mixed_value_kind is not None:
            mixed_kind = preprocessor.mixed_value_kinds.get(column.name, column.mixed_value_kind)
            encoding, _, _ = split_mixed_column(
                column_name=column.name,
                series=pd.Series([np.nan] * len(df_model), index=df_model.index),
                source_kind=mixed_kind,
            )
            state_levels = preprocessor.mixed_state_levels.get(column.name, [])
            codes = df_model[encoding.state_column].round().astype("Int64")
            state_values = [
                state_levels[int(code)] if pd.notna(code) and 0 <= int(code) < len(state_levels) else np.nan
                for code in codes
            ]
            restored_columns[column.name] = pd.Series(state_values, index=df_model.index, dtype="object")
            value_mask = restored_columns[column.name].eq(VALUE_STATE_TOKEN)
            value_series = df_model[encoding.value_column]
            if mixed_kind == "numeric":
                restored_values = pd.to_numeric(value_series, errors="coerce").astype("object")
            elif mixed_kind == "datetime":
                restored_values = numeric_to_datetime(value_series).astype("object")
            elif mixed_kind == "time":
                restored_values = value_series.apply(seconds_to_time).astype("object")
            else:
                restored_values = value_series.astype("object")
            restored_columns[column.name] = restored_columns[column.name].where(~value_mask, restored_values)
            restored_columns[column.name] = restored_columns[column.name].where(
                ~pd.Series(state_values, index=df_model.index).eq(MISSING_STATE_TOKEN),
                np.nan,
            )
            continue

        encoded = df_model[column.name]

        if column.kind == "numeric":
            restored_columns[column.name] = pd.to_numeric(encoded, errors="coerce")
            continue

        if column.kind == "datetime":
            restored_columns[column.name] = numeric_to_datetime(encoded)
            continue

        if column.kind == "time":
            restored_columns[column.name] = encoded.apply(seconds_to_time)
            continue

        levels = preprocessor.categorical_levels.get(column.name, [])
        codes = encoded.round().astype("Int64")
        values = [
            levels[int(code)] if pd.notna(code) and 0 <= int(code) < len(levels) else np.nan
            for code in codes
        ]
        restored_columns[column.name] = values

    restored = pd.DataFrame(restored_columns, index=df_model.index)
    return restored[preprocessor.schema.column_order]


def _safe_fill_value(series: pd.Series) -> float:
    median = series.median(skipna=True)
    if pd.isna(median):
        return 0.0
    return float(median)
