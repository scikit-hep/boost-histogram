// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#include <bh_python/register_histogram.hpp>
#include <bh_python/register_histograms.hpp>

void register_histogram_mean(py::module& hist) {
    register_histogram<storage::mean>(
        hist,
        "any_mean",
        "N-dimensional histogram for sampled data with any axis types.");
}

void register_histogram_weighted_mean(py::module& hist) {
    register_histogram<storage::weighted_mean>(
        hist,
        "any_weighted_mean",
        "N-dimensional histogram for weighted and sampled data with any axis types.");
}
