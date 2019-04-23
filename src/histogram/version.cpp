// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#include <boost/histogram/python/pybind11.hpp>

#include <boost/histogram/histogram.hpp>

void register_version(py::module &m) {
    m.attr("BOOST_HISTOGRAM_DETAIL_AXES_LIMIT") = BOOST_HISTOGRAM_DETAIL_AXES_LIMIT;

    py::module ver = py::module::import("boost.histogram_version");

    m.attr("__version__") = ver.attr("__version__");
}
