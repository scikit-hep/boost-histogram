// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <bh_python/pybind11.hpp>

#include <pybind11/pytypes.h>

struct metadata_t : py::dict {
    PYBIND11_OBJECT(metadata_t, dict, PyDict_Check);

    using dict::dict;

    // default initialize to empty dict (must not be explicit)
    metadata_t()
        : dict() {}

    bool operator==(const metadata_t& other) const { return py::dict::equal(other); }
    bool operator!=(const metadata_t& other) const {
        return py::dict::not_equal(other);
    }
};
