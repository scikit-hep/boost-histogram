// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#include <boost/histogram/python/pybind11.hpp>

#include <boost/histogram/axis/polymorphic_bin.hpp>
#include <boost/histogram/python/bin_setup.hpp>

void register_polymorphic_bin(py::module &ax) {
    py::class_<bh::axis::polymorphic_bin<double>> poly(ax, "polymorphic_bin");

    bin_setup(poly);
}
