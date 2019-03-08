// Copyright 2015-2018 Hans Dembinski and Henry Schreiner
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

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
    
    py::class_<bh::unlimited_storage<>>(storage, "unlimited", "Default adaptive storage")
    .def(py::init<>(), "Default constructor")
    ;
    
    py::class_<vector_int_storage>(storage, "vector", "Integers in vectors storage type")
    .def(py::init<>(), "Default constructor")
    ;
    
    py::class_<bh::weight_storage>(storage, "weight", "Weighted storage type")
    .def(py::init<>(), "Default constructor")
    ;
    
    
}
