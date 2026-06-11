// Copyright 2018-2019 Hans Dembinski and Henry Schreiner
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <bh_python/pybind11.hpp>

#include <boost/core/span.hpp>

/// Generate empty C-contiguous array with same shape as argument
///
/// Note: the result is always C-contiguous regardless of the input's memory
/// layout (strides are *not* copied); callers fill it linearly in C order from
/// a C-contiguous copy of the input.
template <class T>
py::array_t<T> array_like(const py::object& obj) {
    if(!py::isinstance<py::array>(obj)) {
        py::ssize_t shape[1] = {0}; // if scalar
        if(py::isinstance<py::sequence>(obj) && !py::isinstance<py::str>(obj)) {
            // if sequence
            auto seq = py::cast<py::sequence>(obj);
            shape[0] = static_cast<py::ssize_t>(seq.size());
        }
        return py::array_t<T>(shape);
    }
    auto arr = py::cast<py::array>(obj);
    return py::array_t<T>{boost::span<const py::ssize_t>{
        arr.shape(), static_cast<std::size_t>(arr.ndim())}};
}
