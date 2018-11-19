// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "getpython.hpp"

#include <boost/histogram/axis/ostream_operators.hpp>
#include <boost/histogram.hpp>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace bh = boost::histogram;

void register_axis(py::module &m) {

    py::class_<bh::axis::regular<>>(m, "Construct n bins over real range [begin, end).\n"
                                    "\n"
                                    "* n        number of bins.\n"
                                    "* start    low edge of first bin.\n"
                                    "* stop     high edge of last bin.\n"
                                    "* metadata description of the axis. (NA)\n"
                                    "* options  extra bin options. (NA)")
    .def(py::init<unsigned, double, double>(),
         "n"_a, "start"_a, "stop"_a)

    ;

}
