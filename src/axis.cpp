// Copyright 2015-2018 Hans Dembinski and Henry Schreiner
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "getpython.hpp"

#include "axis.hpp"

#include <boost/histogram/axis/ostream_operators.hpp>
#include <boost/histogram.hpp>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace bh = boost::histogram;

using regular_axis = bh::axis::regular<>;
using regular_axes = std::vector<regular_axis>;
using regular_axes_storage = bh::storage_adaptor<regular_axes>;


void register_axis(py::module &m) {

    py::class_<regular_axis>(m, "regular_axis")
    .def(py::init<unsigned, double, double>(),
         "n"_a, "start"_a, "stop"_a)

    ;
    
    py::class_<regular_axes_storage>(m, "regular_axes_storage")
    .def(py::init<regular_axes>(), "Vector of regular axes"_a)
    
    ;

}
