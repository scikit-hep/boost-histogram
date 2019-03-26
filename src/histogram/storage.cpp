// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#include <boost/histogram/python/pybind11.hpp>

#include <boost/histogram/python/storage.hpp>

#include <boost/histogram.hpp>
#include <boost/histogram/storage_adaptor.hpp>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace bh = boost::histogram;

void register_storage(py::module &m) {

    py::module storage = m.def_submodule("storage");

    // Fast storages

    py::class_<dense_int_storage>(storage, "dense_int", "Integers in vectors storage type")
    .def(py::init<>())
    ;

    py::class_<dense_double_storage>(storage, "dense_double", "Weighted storage without variance type (fast but simple)")
    .def(py::init<>())
    ;

    py::class_<dense_atomic_int_storage>(storage, "dense_atomic_int", "Threadsafe (not growing axis) integer storage")
    .def(py::init<>())
    ;

    // Default storages

    py::class_<bh::unlimited_storage<>>(storage, "unlimited", "Pptimized for unweighted histograms, adaptive")
    .def(py::init<>())
    ;

    py::class_<bh::weight_storage>(storage, "weight", "Dense storage which tracks sums of weights and a variance estimate")
    .def(py::init<>())
    ;

    py::class_<bh::profile_storage>(storage, "profile", "Dense storage which tracks means of samples in each cell")
    .def(py::init<>())
    ;

    py::class_<bh::weighted_profile_storage>(storage, "weighted_profile", "Dense storage which tracks means of weighted samples in each cell")
    .def(py::init<>())
    ;

}

any_storage_variant extract_storage(py::kwargs kwargs) {

    if(kwargs.contains("storage")) {
        try {
            return py::cast<dense_int_storage>(kwargs["storage"]);
        } catch(const py::cast_error&) {}

        try {
            return py::cast<dense_double_storage>(kwargs["storage"]);
        } catch(const py::cast_error&) {}

        try {
            return py::cast<bh::unlimited_storage<>>(kwargs["storage"]);
        } catch(const py::cast_error&) {}

        try {
            return py::cast<bh::weight_storage>(kwargs["storage"]);
        } catch(const py::cast_error&) {}

        try {
            return py::cast<dense_atomic_int_storage>(kwargs["storage"]);
        } catch(const py::cast_error&) {}

        throw std::runtime_error("Storage type not supported");

    } else {
        // Default storage if not is specified
        return dense_int_storage();
    }
}
