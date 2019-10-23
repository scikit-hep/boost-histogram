// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <boost/histogram/python/pybind11.hpp>

#include <boost/histogram/python/serializion.hpp>

#include <pybind11/operators.h>
#include <utility>

template <class A, class... Args>
py::class_<A> register_accumulator(py::module acc, Args &&... args) {
    return py::class_<A>(acc, std::forward<Args>(args)...)
        .def(py::init<>())

        .def(py::self += py::self)
        .def(py::self == py::self)
        .def(py::self != py::self)

        .def(py::self *= double())

        .def("__repr__", &shift_to_string<A>)

        .def("__copy__", [](const A &self) { return A(self); })
        .def("__deepcopy__", [](const A &self, py::object) { return A(self); })

        .def(make_pickle<A>());
}
