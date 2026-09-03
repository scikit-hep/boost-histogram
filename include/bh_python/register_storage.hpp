// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <bh_python/pybind11.hpp>

#include <bh_python/def_eq.hpp>
#include <bh_python/make_pickle.hpp>
#include <bh_python/storage.hpp>

#include <cstddef>
#include <stdexcept>

/// Add helpers common to all storage types
template <class A>
py::class_<A> register_storage(py::module& m, const char* name, const char* desc) {
    py::class_<A> storage(m, name, desc);
    def_eq(storage);

    storage.def(py::init<>())
        .def(make_pickle<A>())
        .def("__copy__", [](const A& self) { return A(self); })
        .def("__deepcopy__", [](const A& self, const py::object&) { return A(self); });

    return storage;
}

/// Add helpers to the unlimited storage type
template <>
py::class_<storage::unlimited> inline register_storage(py::module& m,
                                                       const char* name,
                                                       const char* desc) {
    using A = storage::unlimited; // match code above

    py::class_<A> storage(m, name, desc);
    def_eq(storage);

    storage.def(py::init<>())
        .def(make_pickle<A>())
        .def("__copy__", [](const A& self) { return A(self); })
        .def("__deepcopy__", [](const A& self, const py::object&) { return A(self); });

    return storage;
}

/// Add helpers to the multi_cell storage type
template <>
py::class_<storage::multi_cell> inline register_storage(py::module& m,
                                                        const char* name,
                                                        const char* desc) {
    using A = storage::multi_cell; // match code above

    py::class_<A> storage(m, name, desc);
    def_eq(storage);

    storage
        .def(py::init([](int k) {
                 if(k < 1)
                     throw std::invalid_argument("MultiCell nelem must be 1 or larger");
                 return A{static_cast<std::size_t>(k)};
             }),
             py::arg("k"))
        .def(make_pickle<A>())
        .def("__copy__", [](const A& self) { return A(self); })
        .def("__deepcopy__", [](const A& self, const py::object&) { return A(self); })
        .def_property_readonly("nelem", [](const A& self) { return self.nelem(); })

        ;

    return storage;
}
