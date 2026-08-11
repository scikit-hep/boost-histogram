// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#include <bh_python/register_histogram.hpp>
#include <bh_python/register_histograms.hpp>

void register_histogram_double(py::module& hist) {
    register_histogram<storage::double_>(
        hist,
        "any_double",
        "N-dimensional histogram for real-valued data with weights with any axis "
        "types.");
}

void register_histogram_unlimited(py::module& hist) {
    register_histogram<storage::unlimited>(
        hist,
        "any_unlimited",
        "N-dimensional histogram for unlimited size data with any axis types.");
}
