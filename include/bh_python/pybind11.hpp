// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

// This file must be the first include (that is, it must be before stdlib or boost
// headers) This reduces the warning output from Python.h and also lets us define s afew
// things

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <functional>
#include <sstream>
#include <type_traits>

namespace py = pybind11;
using namespace pybind11::literals; // For ""_a syntax

namespace boost {
namespace histogram {}
} // namespace boost
namespace bh = boost::histogram;

/// Static if standin: define a method if expression is true
template <class T, class... Args>
void def_optionally(T&& module, std::true_type, Args&&... expression) {
    module.def(std::forward<Args...>(expression...));
}

/// Static if standin: Do nothing if compile time expression is false
template <class T, class... Args>
void def_optionally(T&&, std::false_type, Args&&...) {}

/// Shift to string
template <class T>
std::string shift_to_string(const T& x) {
    std::ostringstream out;
    out << x;
    return out.str();
}

template <class Obj>
void unchecked_set_impl(std::true_type, py::tuple& tup, ssize_t i, Obj&& obj) {
    // PyTuple_SetItem steals a reference to 'val'
    if(PyTuple_SetItem(tup.ptr(), i, obj.release().ptr()) != 0) {
        throw py::error_already_set();
    }
}

template <class T>
void unchecked_set_impl(std::false_type, py::tuple& tup, ssize_t i, T&& t) {
    unchecked_set_impl(std::true_type{}, tup, i, py::cast(std::forward<T>(t)));
}

/// Unchecked tuple assign
template <class T>
void unchecked_set(py::tuple& tup, std::size_t i, T&& t) {
    unchecked_set_impl(std::is_base_of<py::object, std::decay_t<T>>{},
                       tup,
                       static_cast<ssize_t>(i),
                       std::forward<T>(t));
}
