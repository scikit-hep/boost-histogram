from __future__ import annotations

from typing import Any

import h5py
import numpy as np

from .. import Histogram
from .generic import from_dict, to_dict

__all__ = ["read_hdf5_schema", "write_hdf5_schema"]


def __dir__() -> list[str]:
    return __all__


def write_hdf5_schema(grp: h5py.Group, /, histogram: Histogram) -> None:
    """
    Write a histogram to an HDF5 group.
    """
    hist_dict = to_dict(histogram)

    # All referenced objects will be stored inside of /{name}/ref_axes
    hist_folder_storage = grp.create_group("ref_axes")

    # Metadata

    if "metadata" in hist_dict:
        metadata_grp = grp.create_group("metadata")
        for key, value in hist_dict["metadata"].items():
            metadata_grp.attrs[key] = value

    # Axes
    axes_dataset = grp.create_dataset(
        "axes", len(hist_dict["axes"]), dtype=h5py.special_dtype(ref=h5py.Reference)
    )
    for i, axis in enumerate(hist_dict["axes"]):
        # Iterating through the axes, calling `create_axes_object` for each of them,
        # creating references to new groups and appending it to the `items` dataset defined above
        ax_group = hist_folder_storage.create_group(f"axis_{i}")
        ax_info = axis.copy()
        ax_metadata = ax_info.pop("metadata", None)
        ax_edges = ax_info.pop("edges", None)
        ax_cats = ax_info.pop("items", None)
        for key, value in ax_info.items():
            ax_group.attrs[key] = value
        if ax_metadata is not None:
            ax_metadata_grp = ax_group.create_group("metadata")
            for k, v in ax_metadata.items():
                ax_metadata_grp.attrs[k] = v
        if ax_edges is not None:
            ax_group.create_dataset("edges", shape=ax_edges.shape, data=ax_edges)
        if ax_cats is not None:
            ax_group.create_dataset("items", shape=ax_cats.shape, data=ax_cats)
        axes_dataset[i] = ax_group.ref

    # Storage
    storage_grp = grp.create_group("storage")
    storage_type = hist_dict["storage"]["type"]

    storage_grp.attrs["type"] = storage_type

    for key, value in hist_dict["storage"].items():
        if key == "type":
            continue
        storage_grp.create_dataset(key, shape=value.shape, data=value)


def read_hdf5_schema(grp: h5py.Group, /) -> Histogram:
    """
    Read a histogram from an HDF5 group.
    """
    axes_grp = grp["axes"]
    axes_ref = grp["ref_axes"]
    assert isinstance(axes_ref, h5py.Group)
    assert isinstance(axes_grp, h5py.Dataset)

    axes = [dict(axes_ref[unref_axis_ref].attrs) for unref_axis_ref in axes_ref]

    storage_grp = grp["storage"]
    assert isinstance(storage_grp, h5py.Group)
    storage: dict[str, Any] = {"type": storage_grp.attrs["type"]}
    for key in storage_grp:
        storage[key] = np.asarray(storage_grp[key])

    histogram_dict = {"axes": axes, "storage": storage}
    if "metadata" in grp:
        histogram_dict["metadata"] = dict(grp["metadata"].attrs.items())

    return from_dict(histogram_dict)
