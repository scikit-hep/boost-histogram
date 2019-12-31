// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <bh_python/pybind11.hpp>

/// Get and remove a value from a keyword argument dict
template <class T = py::object>
T required_arg(py::kwargs& kwargs, const char* name) {
    if(kwargs.contains(name)) {
        return py::cast<T>(kwargs.attr("pop")(name));
    } else {
        throw py::key_error(std::string(name) + " is required");
    }
}

/// Get and remove a value from a keyword argument dict or return None
inline py::object optional_arg(py::kwargs& kwargs, const char* name) {
    if(kwargs.contains(name)) {
        return kwargs.attr("pop")(name);
    } else {
        return py::none();
    }
}

/// Get and remove a value from a keyword argument dict or return default value
template <class T>
T optional_arg(py::kwargs& kwargs, const char* name, T original_value) {
    if(kwargs.contains(name)) {
        return py::cast<T>(kwargs.attr("pop")(name));
    } else {
        return original_value;
    }
}

/// Run this last; it will provide an error if other keyword are still present
inline void finalize_args(const py::kwargs& kwargs) {
    if(kwargs.size() > 0) {
        auto keys = py::str(", ").attr("join")(kwargs.attr("keys")());
        throw py::key_error(py::str("Unidentified keywords found: {0}").format(keys));
    }
}
