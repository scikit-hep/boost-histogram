// Copyright 2018-2019 Hans Dembinski and Henry Schreiner
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <bh_python/pybind11.hpp>

#include <boost/histogram/detail/span.hpp>
#include <vector>

/// Generate empty array with same shape and strides as argument
template <class T>
py::array_t<T> array_like(py::object obj) {
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
    std::vector<py::ssize_t> strides;
    strides.reserve(static_cast<std::size_t>(arr.ndim()));
    for(int i = 0; i < arr.ndim(); ++i) {
        strides.emplace_back(arr.strides()[i] / arr.itemsize()
                             * static_cast<py::ssize_t>(sizeof(T)));
    }
    return py::array_t<T>{bh::detail::span<const py::ssize_t>{
                              arr.shape(), static_cast<std::size_t>(arr.ndim())},
                          strides};
}
