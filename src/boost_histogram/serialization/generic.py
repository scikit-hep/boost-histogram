from __future__ import annotations

from .. import storage

__all__ = ["AXIS_MAP", "STORAGE_MAP", "STORAGE_TYPES"]


def __dir__() -> list[str]:
    return __all__


AXIS_MAP = {
    "Regular": "regular",
    "Variable": "variable",
    "IntCategory": "category_int",
    "StrCategory": "category_str",
    "Boolean": "boolean",
}

STORAGE_MAP = {
    "Int64": "int_storage",
    "Double": "double_storage",
    "Weight": "weighted_storage",
    "Mean": "mean_storage",
    "WeightedMean": "weighted_mean_storage",
}

STORAGE_TYPES: dict[str, storage.Storage] = {
    "Int64": storage.Int64(),
    "Double": storage.Double(),
    "Weight": storage.Weight(),
    "Mean": storage.Mean(),
    "WeightedMean": storage.WeightedMean(),
}
