// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#include <bh_python/pybind11.hpp>

#include <bh_python/register_histograms.hpp>

#include <boost/histogram/detail/axes.hpp>

// NOLINTNEXTLINE(misc-use-internal-linkage)
void register_histograms(py::module& hist) {
    hist.attr("_axes_limit") = BOOST_HISTOGRAM_DETAIL_AXES_LIMIT;

    register_histogram_int64(hist);
    register_histogram_unlimited(hist);
    register_histogram_double(hist);
    register_histogram_atomic_int64(hist);
    register_histogram_weight(hist);
    register_histogram_mean(hist);
    register_histogram_weighted_mean(hist);
    register_histogram_multi_cell(hist);
}
