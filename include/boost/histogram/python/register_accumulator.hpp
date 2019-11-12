// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <boost/histogram/python/pybind11.hpp>

#include <boost/histogram/python/accumulators/ostream.hpp>
#include <boost/histogram/python/make_pickle.hpp>

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

        .def("__repr__",
             [](py::object self) {
                 const A &item = py::cast<const A &>(self);
                 return py::str("{0}{1}").format(
                     self.attr("__class__").attr("__name__"), shift_to_string(item));
             })

        .def("__copy__", [](const A &self) { return A(self); })
        .def("__deepcopy__", [](const A &self, py::object) { return A(self); })

        .def(make_pickle<A>());
}
