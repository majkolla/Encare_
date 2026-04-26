from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ColumnSchema:
    name: str
    kind: str
    pandas_dtype: str
    nullable: bool
    unique_count: int
    missing_rate: float
    numeric_like: bool = False
    low_cardinality: bool = False
    id_like: bool = False
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: list[Any] = field(default_factory=list)
    mixed_value_kind: str | None = None
    mixed_token_values: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Schema:
    columns: list[ColumnSchema]
    column_order: list[str]
    row_count: int

    def get(self, column_name: str) -> ColumnSchema:
        for column in self.columns:
            if column.name == column_name:
                return column
        raise KeyError(f"Unknown column: {column_name}")

    @property
    def numeric_columns(self) -> list[str]:
        return [column.name for column in self.columns if column.kind == "numeric"]

    @property
    def categorical_columns(self) -> list[str]:
        return [
            column.name
            for column in self.columns
            if column.kind in {"categorical", "binary", "id_like"} and column.mixed_value_kind is None
        ]

    @property
    def datetime_columns(self) -> list[str]:
        return [column.name for column in self.columns if column.kind == "datetime"]

    @property
    def time_columns(self) -> list[str]:
        return [column.name for column in self.columns if column.kind == "time"]

    @property
    def mixed_columns(self) -> list[str]:
        return [column.name for column in self.columns if column.mixed_value_kind is not None]

    @property
    def modeled_columns(self) -> list[str]:
        return [column.name for column in self.columns if column.kind != "constant"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": [column.to_dict() for column in self.columns],
            "column_order": list(self.column_order),
            "row_count": self.row_count,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Schema":
        return cls(
            columns=[ColumnSchema(**column) for column in payload["columns"]],
            column_order=list(payload["column_order"]),
            row_count=int(payload["row_count"]),
        )


@dataclass
class Preprocessor:
    schema: Schema
    categorical_levels: dict[str, list[Any]]
    numeric_fill_values: dict[str, float]
    datetime_fill_values: dict[str, float]
    time_fill_values: dict[str, float]
    mixed_state_levels: dict[str, list[Any]] = field(default_factory=dict)
    mixed_value_fill_values: dict[str, float] = field(default_factory=dict)
    mixed_value_kinds: dict[str, str] = field(default_factory=dict)
