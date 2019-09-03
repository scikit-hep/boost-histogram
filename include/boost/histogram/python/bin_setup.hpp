// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <boost/histogram/python/pybind11.hpp>

#include <pybind11/operators.h>
#include <string>

using namespace std::literals;

template <class T>
void bin_setup(py::class_<T> &b) {
    b.def("upper", &T::upper)
        .def("lower", &T::lower)
        .def("center", &T::center)
        .def("width", &T::width)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("__repr__",
             [](const T &self) {
                 return "<bin ["s + std::to_string(self.lower()) + ", "s + std::to_string(self.upper()) + "]>"s;
             })

        ;
}
