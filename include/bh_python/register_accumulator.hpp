// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <bh_python/pybind11.hpp>

#include <bh_python/accumulators/ostream.hpp>
#include <bh_python/make_pickle.hpp>

#include <utility>

template <class A, class... Args>
py::class_<A> register_accumulator(py::module acc, Args&&... args) {
    return py::class_<A>(acc, std::forward<Args>(args)...)
        .def(py::init<>())

        .def(py::self += py::self)
        .def(py::self == py::self)
        .def(py::self != py::self)

        .def(py::self *= double())

        // The c++ name is replaced with the Python name here
        .def("__repr__",
             [](py::object self) {
                 const A& item = py::cast<const A&>(self);
                 py::str str   = shift_to_string(item);
                 return py::str("{0.__class__.__name__}({1})").format(self, str);
             })

        .def("__copy__", [](const A& self) { return A(self); })
        .def("__deepcopy__", [](const A& self, py::object) { return A(self); })

        .def(make_pickle<A>());
}
