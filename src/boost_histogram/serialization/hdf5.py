from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .. import Histogram
from .. import axis as baxis
from ..view import WeightedMeanView
from .generic import AXIS_MAP, STORAGE_MAP, STORAGE_TYPES

__all__ = ["read_hdf5_schema", "write_hdf5_schema"]


def __dir__() -> list[str]:
    return __all__


def write_hdf5_schema(f: h5py.File, /, histograms: dict[str, Histogram]) -> None:
    for name, histogram in histograms.items():
        # All referenced objects will be stored inside of /{name}/ref_storage
        f.create_group(f"{name}")
        group_prefix = f"/{name}"
        f[group_prefix].create_group("ref_storage")

        # Metadata

        f[group_prefix].create_group("metadata")
        f[f"{group_prefix}/metadata"].attrs["description"] = (
            "Arbitrary metadata dictionary."
        )
        if histogram.metadata is not None:
            for key, value in histogram.metadata.items():
                f[f"{group_prefix}/metadata"].attrs[key] = value

        # Axes

        f[group_prefix].create_group("axes")
        f[f"{group_prefix}/axes"].attrs["description"] = (
            "A list of the axes of the histogram."
        )
        f[f"{group_prefix}/axes"].create_dataset(
            "items", len(histogram.axes), dtype=h5py.special_dtype(ref=h5py.Reference)
        )
        for i, axis in enumerate(histogram.axes):
            # Iterating through the axes, calling `create_axes_object` for each of them,
            # creating references to new groups and appending it to the `items` dataset defined above

            current_axis = AXIS_MAP[str(axis)[: str(axis).index("(")]]
            dataset = f[f"{group_prefix}/axes/items"]
            args_dict: dict[str, object] = {}
            if current_axis == "regular":
                args_dict["bins"] = len(axis.edges) - 1
                args_dict["lower"] = axis.edges[0]
                args_dict["upper"] = axis.edges[-1]
                args_dict["underflow"] = axis.traits.underflow
                args_dict["overflow"] = axis.traits.overflow
                args_dict["circular"] = axis.traits.circular
            elif current_axis == "variable":
                args_dict["edges"] = axis.edges
                args_dict["underflow"] = axis.traits.underflow
                args_dict["overflow"] = axis.traits.overflow
                args_dict["circular"] = axis.traits.circular
            elif current_axis == "boolean":
                # NOTE: Boolean axes may only have `metadata` as user-input options
                pass
            elif current_axis in {"category_int", "category_str"}:
                s = str(axis)
                args_dict["items"] = np.array(
                    ast.literal_eval(s[s.find("[") : s.find("]") + 1]), dtype=object
                )
                args_dict["flow"] = axis.traits.growth

            if axis.metadata is not None:
                args_dict["metadata"] = axis.metadata
                dataset[i] = create_axes_object(
                    current_axis, f, name, i, True, args_dict
                )[1]
            else:
                dataset[i] = create_axes_object(
                    current_axis, f, name, i, False, args_dict
                )[1]

        # Storage

        f[group_prefix].create_group("storage")
        f[f"{group_prefix}/storage"].attrs["description"] = (
            "The storage of the bins of the histogram."
        )
        hist_str_type = str(histogram.storage_type)
        hist_str_type = STORAGE_MAP[
            hist_str_type[hist_str_type.find("e") + 2 : len(hist_str_type) - 2]
        ]
        args_dict = {}
        args_dict["values"] = np.array(histogram.values())
        if hist_str_type == "int_storage":
            # NOTE: `int_storage` only has the stored values
            pass
        elif hist_str_type == "double_storage":
            # NOTE: `double_storage` only has the stored values
            pass
        elif hist_str_type == "weighted_storage":
            args_dict["variances"] = np.array(histogram.variances())
        elif hist_str_type == "mean_storage":
            args_dict["variances"] = np.array(histogram.variances())
            args_dict["counts"] = np.array(histogram.counts())
        elif hist_str_type == "weighted_mean_storage":
            view = histogram.view()
            assert isinstance(view, WeightedMeanView)
            args_dict["variances"] = np.array(histogram.variances())
            args_dict["sum_of_weights"] = view.sum_of_weights
            args_dict["sum_of_weights_squared"] = view.sum_of_weights_squared

        create_storage_object(hist_str_type, f, name, args_dict)


def read_hdf5_schema(input_file: h5py.File | Path) -> dict[str, Histogram]:
    f = h5py.File(input_file) if isinstance(input_file, Path) else input_file
    op_dict = {}
    # The first level in the schema are the various histograms that have been serialized
    for hist_name in list(f.keys()):
        base_prefix = f"/{hist_name}"

        #### `metadata` code start
        metadata = {}
        metadata_ref = f[f"{base_prefix}/metadata"]
        for key, value in metadata_ref.attrs.items():
            metadata[key] = value
        #### `metadata` code end

        #### `axes` code start
        axes: list[baxis.Axis] = []
        axes_ref = f[f"{base_prefix}/axes"]
        for i, unref_axis_ref in enumerate(axes_ref["items"]):
            deref_axis_ref = f[unref_axis_ref]
            axis_type = deref_axis_ref.attrs["type"]
            args_dict: dict[str, Any] = {}
            # HACK: Force-adding the metadata field in `args_dict` allows me to avoid
            # making if-else test around the existence of metadata for the current axis
            args_dict["metadata"] = {}
            for key, value in deref_axis_ref.attrs.items():
                args_dict[key] = value
            if axis_type == "regular":
                axes.append(
                    baxis.Regular(
                        args_dict["bins"],
                        args_dict["lower"],
                        args_dict["upper"],
                        overflow=args_dict["overflow"],
                        underflow=args_dict["underflow"],
                        circular=args_dict["circular"],
                        metadata=args_dict["metadata"],
                    )
                )
            elif axis_type == "variable":
                args_dict["edges"] = np.array(deref_axis_ref[f"axis_{i}_edges"])
                axes.append(
                    baxis.Variable(
                        args_dict["edges"],
                        underflow=args_dict["underflow"],
                        overflow=args_dict["overflow"],
                        circular=args_dict["circular"],
                        metadata=args_dict["metadata"],
                    )
                )
            elif axis_type == "boolean":
                axes.append(baxis.Boolean(metadata=args_dict["metadata"]))
            elif axis_type == "category_int":
                args_dict["items"] = np.array(deref_axis_ref[f"axis_{i}_categories"])
                axes.append(
                    baxis.IntCategory(
                        args_dict["items"],
                        growth=args_dict["flow"],
                        metadata=args_dict["metadata"],
                    )
                )
            elif axis_type == "category_str":
                args_dict["items"] = np.array(deref_axis_ref[f"axis_{i}_categories"])
                axes.append(
                    baxis.StrCategory(
                        args_dict["items"],
                        growth=args_dict["flow"],
                        metadata=args_dict["metadata"],
                    )
                )
        #### `axes` code end

        #### `storage` code start
        storage_ref = f[f"{base_prefix}/storage"]
        storage_type = storage_ref.attrs["type"]
        # NOTE: We construct the corresponding `bh.Histogram` object and assign the values
        # from the serialization directly
        h = Histogram(
            *axes,
            storage=STORAGE_TYPES[{v: k for k, v in STORAGE_MAP.items()}[storage_type]],
        )
        if storage_type in {"int_storage", "double_storage"}:
            h[...] = np.array(storage_ref["data"])
        elif storage_type == "weighted_storage":
            h[...] = np.stack(
                [np.array(storage_ref["data"]), np.array(storage_ref["variances"])],
                axis=-1,
            )
        elif storage_type == "mean_storage":
            h[...] = np.stack(
                [
                    np.array(storage_ref["counts"]),
                    np.array(storage_ref["data"]),
                    np.array(storage_ref["variances"]),
                ],
                axis=-1,
            )
        elif storage_type == "weighted_mean_storage":
            h[...] = np.stack(
                [
                    np.array(storage_ref["sum_of_weights"]),
                    np.array(storage_ref["sum_of_weights_squared"]),
                    np.array(storage_ref["data"]),
                    np.array(storage_ref["variances"]),
                ],
                axis=-1,
            )
        else:
            msg = f"Unknown storage type: {storage_type}"
            raise ValueError(msg)

        op_dict[hist_name] = h
    return op_dict


def create_axes_object(
    axis_type: str,
    hdf5_ptr: h5py.File,
    hist_name: str,
    axis_num: int,
    has_metadata: bool,
    args_dict: dict[str, Any],
) -> tuple[h5py.File, h5py.Reference]:
    """Helper function for constructing and adding a new axis in the /ref_storage subfolder inside
    /hist_name of the hdf5_ptr file"""
    hist_folder_storage = hdf5_ptr[f"/{hist_name}/ref_storage"]
    ref = hist_folder_storage.create_group(f"axis_{axis_num}")
    if axis_type == "regular":
        ref.attrs["type"] = axis_type
        ref.attrs["description"] = "An evenly spaced set of continuous bins."
        ref.attrs["bins"] = args_dict["bins"]
        ref.attrs["lower"] = args_dict["lower"]
        ref.attrs["upper"] = args_dict["upper"]
        ref.attrs["underflow"] = args_dict["underflow"]
        ref.attrs["overflow"] = args_dict["overflow"]
        ref.attrs["circular"] = args_dict["circular"]
        if has_metadata:
            ref.create_group("metadata")
            for key, value in args_dict["metadata"].items():
                ref["/metadata"].attrs[key] = value
    elif axis_type == "variable":
        ref.attrs["type"] = axis_type
        ref.attrs["description"] = "A variably spaced set of continuous bins."
        # HACK: requires `Variable` data is passed in as a
        # numpy array
        ref.create_dataset(
            f"axis_{axis_num}_edges",
            shape=args_dict["edges"].shape,
            data=args_dict["edges"],
        )
        ref.attrs["underflow"] = args_dict["underflow"]
        ref.attrs["overflow"] = args_dict["overflow"]
        ref.attrs["circular"] = args_dict["circular"]
        if has_metadata:
            ref.create_group("metadata")
            for key, value in args_dict["metadata"].items():
                ref["/metadata"].attrs[key] = value
    elif axis_type == "boolean":
        ref.attrs["type"] = axis_type
        ref.attrs["description"] = "A simple true/false axis with no flow."
        if has_metadata:
            ref.create_group("metadata")
            for key, value in args_dict["metadata"].items():
                ref["/metadata"].attrs[key] = value
    elif axis_type == "category_int":
        ref.attrs["type"] = axis_type
        ref.attrs["description"] = "A set of integer categorical bins in any order."
        ref.create_dataset(
            f"axis_{axis_num}_categories",
            shape=args_dict["items"].shape,
            data=args_dict["items"],
        )
        ref.attrs["flow"] = args_dict["flow"]
        if has_metadata:
            ref.create_group("metadata")
            for key, value in args_dict["metadata"].items():
                ref["/metadata"].attrs[key] = value
    elif axis_type == "category_str":
        ref.attrs["type"] = axis_type
        ref.attrs["description"] = "A set of string categorical bins."
        # HACK: Assumes that the input is a numpy array of strings
        # This is typically imposed via `dtype=object`
        ref.create_dataset(
            f"axis_{axis_num}_categories",
            shape=args_dict["items"].shape,
            data=args_dict["items"],
        )
        ref.attrs["flow"] = args_dict["flow"]
        if has_metadata:
            ref.create_group("metadata")
            for key, value in args_dict["metadata"].items():
                ref["/metadata"].attrs[key] = value
    return (hdf5_ptr, ref.ref)


def create_storage_object(
    storage_type: str, hdf5_ptr: h5py.File, hist_name: str, args_dict: dict[str, Any]
) -> h5py.File:
    """Helper function for constructing and storing the main data in the /ref_storage
    subfolder inside /hist_name of the hdf5_ptr file"""
    ref = hdf5_ptr[f"/{hist_name}/storage"]
    ref.attrs["type"] = storage_type
    ref.create_dataset(
        "data", shape=args_dict["values"].shape, data=args_dict["values"]
    )
    # storage_type = STORAGE_MAP[storage_type]
    if storage_type == "int_storage":
        ref.attrs["description"] = "A storage holding integer counts."
    elif storage_type == "double_storage":
        ref.attrs["description"] = "A storage holding floating point counts."
    elif storage_type == "weighted_storage":
        ref.attrs["description"] = (
            "A storage holding floating point counts and variances."
        )
        ref.create_dataset(
            "variances",
            shape=args_dict["variances"].shape,
            data=args_dict["variances"],
        )
    elif storage_type == "mean_storage":
        ref.attrs["description"] = (
            "A storage holding 'profile'-style floating point counts, values, and variances."
        )
        ref.create_dataset(
            "counts", shape=args_dict["counts"].shape, data=args_dict["counts"]
        )
        ref.create_dataset(
            "variances",
            shape=args_dict["variances"].shape,
            data=args_dict["variances"],
        )
    elif storage_type == "weighted_mean_storage":
        ref.attrs["description"] = (
            "A storage holding 'profile'-style floating point ∑weights, ∑weights², values, and variances."
        )
        ref.create_dataset(
            "variances",
            shape=args_dict["variances"].shape,
            data=args_dict["variances"],
        )
        ref.create_dataset(
            "sum_of_weights",
            shape=args_dict["sum_of_weights"].shape,
            data=args_dict["sum_of_weights"],
        )
        ref.create_dataset(
            "sum_of_weights_squared",
            shape=args_dict["sum_of_weights_squared"].shape,
            data=args_dict["sum_of_weights_squared"],
        )
    return hdf5_ptr
