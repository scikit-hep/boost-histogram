// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <bh_python/pybind11.hpp>

#include <bh_python/make_pickle.hpp>
#include <bh_python/storage.hpp>

/// Add helpers common to all storage types
template <class A>
py::class_<A> register_storage(py::module& m, const char* name, const char* desc) {
    py::class_<A> storage(m, name, desc);

    storage.def(py::init<>())
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(make_pickle<A>())
        .def("__copy__", [](const A& self) { return A(self); })
        .def("__deepcopy__", [](const A& self, py::object) { return A(self); });

    return storage;
}

/// Add helpers to the unlimited storage type
template <>
py::class_<storage::unlimited>
register_storage(py::module& m, const char* name, const char* desc) {
    using A = storage::unlimited; // match code above

    py::class_<A> storage(m, name, desc);

    storage.def(py::init<>())
        .def(py::self == py::self)
        .def("__ne__", [](const A& a, const A& b) { return !(a == b); })
        .def(make_pickle<A>())
        .def("__copy__", [](const A& self) { return A(self); })
        .def("__deepcopy__", [](const A& self, py::object) { return A(self); });

    return storage;
}
