from __future__ import annotations

import functools
from typing import Any

from .. import axis

__all__ = ["_axis_from_dict", "_axis_to_dict"]


def __dir__() -> list[str]:
    return __all__


@functools.singledispatch
def _axis_to_dict(ax: Any, /) -> dict[str, Any]:
    """Convert an axis to a dictionary."""
    raise TypeError(f"Unsupported axis type: {type(ax)}")


@_axis_to_dict.register(axis.Regular)
@_axis_to_dict.register(axis.Integer)
def _(ax: axis.Regular | axis.Integer, /) -> dict[str, Any]:
    """Convert a Regular axis to a dictionary."""

    # Special handling if the axis has a transform
    if isinstance(ax, axis.Regular) and ax.transform is not None:
        data = {
            "type": "variable",
            "edges": ax.edges,
            "underflow": ax.traits.underflow,
            "overflow": ax.traits.overflow,
            "circular": ax.traits.circular,
        }
    else:
        data = {
            "type": "regular",
            "lower": ax.edges[0],
            "upper": ax.edges[-1],
            "bins": ax.size,
            "underflow": ax.traits.underflow,
            "overflow": ax.traits.overflow,
            "circular": ax.traits.circular,
        }
    if ax.metadata is not None:
        data["metadata"] = ax.metadata

    return data


@_axis_to_dict.register
def _(ax: axis.Variable, /) -> dict[str, Any]:
    """Convert a Variable or Integer axis to a dictionary."""
    data = {
        "type": "variable",
        "edges": ax.edges,
        "underflow": ax.traits.underflow,
        "overflow": ax.traits.overflow,
        "circular": ax.traits.circular,
    }
    if ax.metadata is not None:
        data["metadata"] = ax.metadata

    return data


@_axis_to_dict.register
def _(ax: axis.IntCategory, /) -> dict[str, Any]:
    """Convert an IntCategory axis to a dictionary."""
    data = {
        "type": "category_int",
        "categories": list(ax),
        "flow": ax.traits.overflow,
    }
    if ax.metadata is not None:
        data["metadata"] = ax.metadata

    return data


@_axis_to_dict.register
def _(ax: axis.StrCategory, /) -> dict[str, Any]:
    """Convert a StrCategory axis to a dictionary."""
    data = {
        "type": "category_str",
        "categories": list(ax),
        "flow": ax.traits.overflow,
    }
    if ax.metadata is not None:
        data["metadata"] = ax.metadata

    return data


@_axis_to_dict.register
def _(ax: axis.Boolean, /) -> dict[str, Any]:
    """Convert a Boolean axis to a dictionary."""
    data = {
        "type": "boolean",
    }
    if ax.metadata is not None:
        data["metadata"] = ax.metadata

    return data


def _axis_from_dict(data: dict[str, Any], /) -> axis.Axis:
    hist_type = data["type"]
    if hist_type == "regular":
        return axis.Regular(
            data["bins"],
            data["lower"],
            data["upper"],
            underflow=data["underflow"],
            overflow=data["overflow"],
            circular=data["circular"],
            metadata=data.get("metadata"),
        )
    if hist_type == "variable":
        return axis.Variable(
            data["edges"],
            underflow=data["underflow"],
            overflow=data["overflow"],
            circular=data["circular"],
            metadata=data.get("metadata"),
        )
    if hist_type == "category_int":
        return axis.IntCategory(
            data["categories"],
            overflow=data["flow"],
            metadata=data.get("metadata"),
        )
    if hist_type == "category_str":
        return axis.StrCategory(
            data["categories"],
            overflow=data["flow"],
            metadata=data.get("metadata"),
        )
    if hist_type == "boolean":
        return axis.Boolean(metadata=data.get("metadata"))

    raise TypeError(f"Unsupported axis type: {hist_type}")
