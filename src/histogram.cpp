// Copyright 2015-2018 Hans Dembinski and Henry Schreiner
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "getpython.hpp"

#include "axis.hpp"

#include <boost/histogram.hpp>
#include <boost/histogram/axis/ostream_operators.hpp>
#include <boost/histogram/ostream_operators.hpp>

#include <cassert>
#include <vector>
#include <sstream>

#ifndef BOOST_HISTOGRAM_AXIS_LIMIT
#define BOOST_HISTOGRAM_AXIS_LIMIT 16
#endif

namespace bh = boost::histogram;

using regular_histogram_t = bh::histogram<regular_axes_storage, bh::default_storage>;

void register_histogram(py::module& m) {
    
    py::class_<regular_histogram_t>(m, "regular_histogram", "N-dimensional histogram for real-valued data.")
    .def(py::init<const regular_axes_storage &, bh::default_storage>(), "axes"_a, "storage"_a=bh::default_storage())
    
    .def("rank", &regular_histogram_t::rank,
         "Number of axes (dimensions) of histogram" )
    .def("size", &regular_histogram_t::size,
         "Total number of bins in the histogram (including underflow/overflow)" )
    .def("reset", &regular_histogram_t::reset,
         "Reset bin counters to zero")
    
    //.def("axis", py::overload_cast<int>(&regular_histogram_t::axis),
    //     "Get N-th axis with runtime index")
    
    .def("__call__", [](regular_histogram_t &self, py::args &args){
        size_t size = args.size();
        if(size == 1)
            self(py::cast<double>(args[0]));
        else if (size == 2)
            self(py::cast<double>(args[0]), py::cast<double>(args[1]));
        else
            throw py::index_error();
        },
        "Add a value to the historgram")
    
    .def("at", [](regular_histogram_t &self, py::args &args){
        size_t size = args.size();
        if(size == 1)
            return self.at(py::cast<int>(args[0]));
        else if (size == 2)
            return self.at(py::cast<int>(args[0]), py::cast<int>(args[1]));
        else
            throw py::index_error();
    },
         "Access bin counter at indices")
    
    .def("__repr__", [](regular_histogram_t &self){
        std::ostringstream out;
        out << self;
        return out.str();
    })
    
    
    ;
}
