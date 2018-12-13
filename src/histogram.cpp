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

using regular_histogram_t = bh::histogram<bh::storage_adaptor<std::vector<bh::axis::regular<>>>>;

void register_histogram(py::module& m) {
    
    py::class_<regular_histogram_t>(m, "regular_histogram", "N-dimensional histogram for real-valued data.");
}
