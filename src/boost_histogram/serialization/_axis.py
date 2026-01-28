from __future__ import annotations

import functools
from typing import Any

from .. import axis
from ._common import serialize_metadata

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

    shared = {
        "underflow": ax.traits.underflow,
        "overflow": ax.traits.overflow,
        "circular": ax.traits.circular,
    }

    # Special handling if the axis has a transform
    if isinstance(ax, axis.Regular) and ax.transform is not None:
        data = {
            "type": "variable",
            "edges": ax.edges,
            **shared,
        }
    elif isinstance(ax, axis.Integer):
        data = {
            "type": "regular",
            "lower": int(ax.edges[0]),
            "upper": int(ax.edges[-1]),
            "bins": ax.size,
            **shared,
        }
    else:
        data = {
            "type": "regular",
            "lower": float(ax.edges[0]),
            "upper": float(ax.edges[-1]),
            "bins": ax.size,
            **shared,
        }

    writer_info = dict[str, str | bool]()
    if isinstance(ax, axis.Integer):
        writer_info["orig_type"] = "Integer"
    if ax.traits.growth:
        writer_info["growth"] = True
    if writer_info:
        data["writer_info"] = {"boost-histogram": writer_info}

    metadata = serialize_metadata(ax.__dict__)
    if metadata:
        data["metadata"] = metadata

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

    if ax.traits.growth:
        data["writer_info"] = {"boost-histogram": {"growth": True}}

    metadata = serialize_metadata(ax.__dict__)
    if metadata:
        data["metadata"] = metadata

    return data


@_axis_to_dict.register
def _(ax: axis.IntCategory, /) -> dict[str, Any]:
    """Convert an IntCategory axis to a dictionary."""
    data = {
        "type": "category_int",
        "categories": list(ax),
        "flow": ax.traits.overflow,
    }

    if ax.traits.growth:
        data["writer_info"] = {"boost-histogram": {"growth": True}}

    metadata = serialize_metadata(ax.__dict__)
    if metadata:
        data["metadata"] = metadata

    return data


@_axis_to_dict.register
def _(ax: axis.StrCategory, /) -> dict[str, Any]:
    """Convert a StrCategory axis to a dictionary."""
    data = {
        "type": "category_str",
        "categories": list(ax),
        "flow": ax.traits.overflow,
    }

    if ax.traits.growth:
        data["writer_info"] = {"boost-histogram": {"growth": True}}

    metadata = serialize_metadata(ax.__dict__)
    if metadata:
        data["metadata"] = metadata

    return data


@_axis_to_dict.register
def _(ax: axis.Boolean, /) -> dict[str, Any]:
    """Convert a Boolean axis to a dictionary."""
    data: dict[str, Any] = {
        "type": "boolean",
    }

    metadata = serialize_metadata(ax.__dict__)
    if metadata:
        data["metadata"] = metadata

    return data


def _axis_from_dict(data: dict[str, Any], /) -> axis.Axis:
    writer_info = data.get("writer_info", {}).get("boost-histogram", {})
    orig_type = writer_info.get("orig_type", "")
    if orig_type == "Integer":
        assert data["upper"] - data["lower"] == data["bins"]
        return axis.Integer(
            data["lower"],
            data["upper"],
            underflow=data["underflow"],
            overflow=data["overflow"],
            circular=data["circular"],
            growth=writer_info.get("growth", False),
            __dict__=data.get("metadata"),
        )

    hist_type = data["type"]
    if hist_type == "regular":
        return axis.Regular(
            data["bins"],
            data["lower"],
            data["upper"],
            underflow=data["underflow"],
            overflow=data["overflow"],
            circular=data["circular"],
            growth=writer_info.get("growth", False),
            __dict__=data.get("metadata"),
        )
    if hist_type == "variable":
        return axis.Variable(
            data["edges"],
            underflow=data["underflow"],
            overflow=data["overflow"],
            circular=data["circular"],
            growth=writer_info.get("growth", False),
            __dict__=data.get("metadata"),
        )
    if hist_type == "category_int":
        return axis.IntCategory(
            data["categories"],
            overflow=data["flow"],
            growth=writer_info.get("growth", False),
            __dict__=data.get("metadata"),
        )
    if hist_type == "category_str":
        return axis.StrCategory(
            data["categories"],
            overflow=data["flow"],
            growth=writer_info.get("growth", False),
            __dict__=data.get("metadata"),
        )
    if hist_type == "boolean":
        return axis.Boolean(__dict__=data.get("metadata"))

    raise TypeError(f"Unsupported axis type: {hist_type}")
