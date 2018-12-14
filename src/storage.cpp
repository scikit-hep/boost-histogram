// Copyright 2015-2018 Hans Dembinski and Henry Schreiner
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "getpython.hpp"

#include <boost/histogram.hpp>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace bh = boost::histogram;

void register_storage(py::module &m) {
    
    py::class_<bh::default_storage>(m, "default_storage", "Default adaptive storage")
    .def(py::init<>(), "Default constructor")
    
    ;
    
}
