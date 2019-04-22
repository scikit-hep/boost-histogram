// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <boost/histogram/python/pybind11.hpp>

#include <boost/histogram/python/pickle.hpp>
#include <boost/histogram/python/storage.hpp>
#include <pybind11/operators.h>

/// Add helpers common to all storage types
template <typename A, typename T>
py::class_<A> register_storage(py::module &m, const char *name, const char *desc) {
    py::class_<A> storage(m, name, desc);

    storage.def(py::init<>())
        .def("__getitem__", [](A &self, size_t ind) { return self.at(ind); })
        .def("__setitem__", [](A &self, size_t ind, T val) { self.at(ind) = val; })
        .def("push_back", [](A &self, T val) { self.push_back(val); })
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(make_pickle<A>())
        .def("__copy__", [](const A &self) { return A(self); })
        .def("__deepcopy__", [](const A &self, py::object) { return A(self); });

    return storage;
}
