// Copyright 2015-2018 Hans Dembinski and Henry Schreiner
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/histogram/python/pybind11.hpp>

#include <boost/histogram/python/axis.hpp>

#include <boost/histogram/axis/ostream.hpp>
#include <boost/histogram.hpp>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace bh = boost::histogram;

void register_axis(py::module &m) {
    
    py::module ax = m.def_submodule("axis");

    // Axis types
    py::class_<axis::regular>(ax, "regular")
    .def(py::init<unsigned, double, double>(), "n"_a, "start"_a, "stop"_a)
    ;
    
    py::class_<axis::circular>(ax, "circular")
    .def(py::init<unsigned, double, double>(), "n"_a, "start"_a, "stop"_a)
    ;
    
    py::class_<axis::regular_log>(ax, "regular_log")
    .def(py::init<unsigned, double, double>(), "n"_a, "start"_a, "stop"_a)
    ;
    
    py::class_<axis::regular_sqrt>(ax, "regular_sqrt")
    .def(py::init<unsigned, double, double>(), "n"_a, "start"_a, "stop"_a)
    ;
    
    py::class_<axis::regular_pow>(ax, "regular_pow")
    .def(py::init([](double pow, unsigned n, double start, double stop){
        return new axis::regular_pow(bh::axis::transform::pow{pow}, n, start , stop);} ), "pow"_a, "n"_a, "start"_a, "stop"_a)
    ;
    
    py::module axs = m.def_submodule("axes");
    
    // Containers of axes
    py::class_<axes::regular>(axs, "regular")
    .def(py::init<axes::regular>(), "Vector of regular axes"_a)
    ;
    
    py::class_<axes::any>(axs, "any")
    .def(py::init<axes::any>(), "Vector of any axes types"_a)
    ;
    

}
