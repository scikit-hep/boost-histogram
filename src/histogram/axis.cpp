// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

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

/// Add helpers common to all axis types
template<typename A>
py::class_<A> register_axis_by_type(py::module& m, const char* name, const char* desc) {
    py::class_<A> axis(m, name, desc);
    
    axis
    .def("__repr__", [](A &self){
        std::ostringstream out;
        out << self;
        return out.str();
    })
    ;
    
    return axis;
}

void register_axis(py::module &m) {
    
    py::module ax = m.def_submodule("axis");

    register_axis_by_type<axis::regular>(ax, "regular", "Evenly spaced bins")
    .def(py::init<unsigned, double, double>(), "n"_a, "start"_a, "stop"_a)
    ;
    
    register_axis_by_type<axis::regular_noflow>(ax, "regular_noflow", "Evenly spaced bins without over/under flow")
    .def(py::init<unsigned, double, double>(), "n"_a, "start"_a, "stop"_a)
    ;
    
    register_axis_by_type<axis::circular>(ax, "circular", "Evenly spaced bins with wraparound")
    .def(py::init<unsigned, double, double>(), "n"_a, "start"_a, "stop"_a)
    ;
    
    register_axis_by_type<axis::regular_log>(ax, "regular_log", "Evenly spaced bins in log10")
    .def(py::init<unsigned, double, double>(), "n"_a, "start"_a, "stop"_a)
    ;
    
    register_axis_by_type<axis::regular_sqrt>(ax, "regular_sqrt", "Evenly spaced bins in sqrt")
    .def(py::init<unsigned, double, double>(), "n"_a, "start"_a, "stop"_a)
    ;
    
    register_axis_by_type<axis::regular_pow>(ax, "regular_pow", "Evenly spaced bins in a power")
    .def(py::init([](double pow, unsigned n, double start, double stop){
        return new axis::regular_pow(bh::axis::transform::pow{pow}, n, start , stop);} ), "pow"_a, "n"_a, "start"_a, "stop"_a)
    ;
    
    register_axis_by_type<axis::variable>(ax, "variable", "Unevenly spaced bins")
    .def(py::init<std::vector<double>>(), "edges"_a)
    ;

    register_axis_by_type<axis::integer>(ax, "integer", "Contigious integers")
    .def(py::init<int, int>(), "min"_a, "max"_a)
    ;

    register_axis_by_type<axis::category_str>(ax, "category_str", "Text label bins")
    .def(py::init<std::vector<std::string>>(), "labels"_a)
    ;
}
