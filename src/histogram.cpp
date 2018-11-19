// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "getpython.hpp"

#include <boost/histogram.hpp>
#include <cassert>
#include <vector>

#ifndef BOOST_HISTOGRAM_AXIS_LIMIT
#define BOOST_HISTOGRAM_AXIS_LIMIT 16
#endif

namespace bh = boost::histogram;

void register_histogram() {
    
    py::class_<bh::histogram<>>(m, "histogram", "N-dimensional histogram for real-valued data.")
        .def(py::init<const axes_type&>(), "axes"_a)
    
    .def("__len__", &bh::histogram<>::,
         ":return: total number of bins, including under- and overflow")
    ;
    
}
