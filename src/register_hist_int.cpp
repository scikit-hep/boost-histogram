// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#include <bh_python/register_histogram.hpp>
#include <bh_python/register_histograms.hpp>

void register_histogram_int64(py::module& hist) {
    register_histogram<storage::int64>(
        hist,
        "any_int64",
        "N-dimensional histogram for unlimited size data with any axis types.");
}

void register_histogram_atomic_int64(py::module& hist) {
    register_histogram<storage::atomic_int64>(
        hist,
        "any_atomic_int64",
        "N-dimensional histogram for threadsafe integer data with any axis types.");
}
