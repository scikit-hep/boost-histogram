// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <bh_python/pybind11.hpp>

#include <pybind11/pytypes.h>

inline bool PyObject_Check(void* value) { return value != nullptr; }

struct metadata_t : py::object {
    PYBIND11_OBJECT(metadata_t, object, PyObject_Check);

    // default initialize to None
    metadata_t()
        : object(Py_None, borrowed_t{}) {}

    bool operator==(const metadata_t& other) const { return py::object::equal(other); }
    bool operator!=(const metadata_t& other) const {
        return py::object::not_equal(other);
    }
};
