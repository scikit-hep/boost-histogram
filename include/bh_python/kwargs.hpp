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

/// Remove a value from a keyword argument dict if None, do not remove if non None
/// (triggers final error)
inline void none_only_arg(py::kwargs& kwargs, const char* name) {
    if(kwargs.contains(name)) {
        if(kwargs[name].is_none())
            kwargs.attr("pop")(name);
    }
}

/// Run this last; it will provide an error if other keyword are still present
inline void finalize_args(const py::kwargs& kwargs) {
    if(!kwargs.empty()) {
        auto keys = py::str(", ").attr("join")(kwargs.attr("keys")());
        throw py::type_error(py::str("Keyword(s) {0} not expected").format(keys));
    }
}
