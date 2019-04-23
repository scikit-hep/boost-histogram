// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#include <boost/histogram/python/pybind11.hpp>

#include <boost/histogram/python/axis.hpp>
#include <boost/histogram/python/histogram.hpp>
#include <boost/histogram/python/register_histogram.hpp>
#include <boost/histogram/python/storage.hpp>

#include <boost/histogram.hpp>

void register_fast_histograms(py::module &hist) {
    register_histogram<axes::regular_uoflow, storage::unlimited>(
        hist, "regular_unlimited", "N-dimensional histogram for real-valued data.");

    register_histogram<axes::regular_uoflow, storage::int_>(
        hist, "regular_int", "N-dimensional histogram for int-valued data.");

    register_histogram<axes::regular_uoflow, storage::atomic_int>(
        hist, "regular_atomic_int", "N-dimensional histogram for atomic int-valued data.");

    register_histogram<axes::regular_noflow, storage::int_>(
        hist, "regular_noflow_int", "N-dimensional histogram for int-valued data.");
}
