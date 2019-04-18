// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#include <boost/histogram/python/pybind11.hpp>

#include <boost/histogram/python/pickle.hpp>
#include <boost/histogram/python/storage.hpp>
#include <pybind11/operators.h>

#include <boost/histogram.hpp>
#include <boost/histogram/storage_adaptor.hpp>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

/// Add helpers common to all storage types
template <typename A, typename T>
py::class_<A> register_storage_by_type(py::module &m, const char *name, const char *desc) {
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

py::module register_storage(py::module &m) {
    py::module storage = m.def_submodule("storage");

    // Fast storages

    register_storage_by_type<storage::int_, unsigned>(storage, "int", "Integers in vectors storage type");

    py::class_<storage::double_>(storage, "double", "Weighted storage without variance type (fast but simple)")
        .def(py::init<>());

    py::class_<storage::atomic_int>(storage, "atomic_int", "Threadsafe (not growing axis) integer storage")
        .def(py::init<>());

    // Default storages

    py::class_<storage::unlimited>(storage, "unlimited", "Optimized for unweighted histograms, adaptive")
        .def(py::init<>());

    py::class_<storage::weight>(storage, "weight", "Dense storage which tracks sums of weights and a variance estimate")
        .def(py::init<>());

    py::class_<storage::profile>(storage, "profile", "Dense storage which tracks means of samples in each cell")
        .def(py::init<>());

    py::class_<storage::weighted_profile>(
        storage, "weighted_profile", "Dense storage which tracks means of weighted samples in each cell")
        .def(py::init<>());

    return storage;
}
