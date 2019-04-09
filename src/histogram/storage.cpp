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


void register_storage(py::module &m) {

    py::module storage = m.def_submodule("storage");

    // Fast storages

    py::class_<storage::int_>(storage, "int", "Integers in vectors storage type")
    .def(py::init<>())
    ;

    py::class_<storage::double_>(storage, "double", "Weighted storage without variance type (fast but simple)")
    .def(py::init<>())
    ;

    py::class_<storage::atomic_int>(storage, "atomic_int", "Threadsafe (not growing axis) integer storage")
    .def(py::init<>())
    ;

    // Default storages

    py::class_<storage::unlimited>(storage, "unlimited", "Optimized for unweighted histograms, adaptive")
    .def(py::init<>())
    ;

    py::class_<storage::weight>(storage, "weight", "Dense storage which tracks sums of weights and a variance estimate")
    .def(py::init<>())
    ;

    py::class_<storage::profile>(storage, "profile", "Dense storage which tracks means of samples in each cell")
    .def(py::init<>())
    ;

    py::class_<storage::weighted_profile>(storage, "weighted_profile", "Dense storage which tracks means of weighted samples in each cell")
    .def(py::init<>())
    ;

}

storage::any_variant extract_storage(py::kwargs kwargs) {

    if(kwargs.contains("storage")) {
        if(py::isinstance<storage::int_>(kwargs["storage"]))
            return py::cast<storage::int_>(kwargs["storage"]);
        else if(py::isinstance<storage::double_>(kwargs["storage"]))
            return py::cast<storage::double_>(kwargs["storage"]);
        else if(py::isinstance<storage::unlimited>(kwargs["storage"]))
            return py::cast<storage::unlimited>(kwargs["storage"]);
        else if(py::isinstance<storage::weight>(kwargs["storage"]))
            return py::cast<storage::weight>(kwargs["storage"]);
        else if(py::isinstance<storage::atomic_int>(kwargs["storage"]))
            return py::cast<storage::atomic_int>(kwargs["storage"]);
        else
            throw std::runtime_error("Storage type not supported");

    } else {
        // Default storage if not is specified
        return storage::int_();
    }
}
