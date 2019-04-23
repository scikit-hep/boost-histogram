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

void register_general_histograms(py::module &hist) {
    register_histogram<axes::any, storage::int_>(
        hist, "any_int", "N-dimensional histogram for int-valued data with any axis types.");

    register_histogram<axes::any, storage::atomic_int>(
        hist, "any_atomic_int", "N-dimensional histogram for int-valued data with any axis types (threadsafe).");

    register_histogram<axes::any, storage::double_>(
        hist, "any_double", "N-dimensional histogram for real-valued data with weights with any axis types.");

    register_histogram<axes::any, storage::unlimited>(
        hist, "any_unlimited", "N-dimensional histogram for unlimited size data with any axis types.");

    register_histogram<axes::any, storage::weight>(
        hist, "any_weight", "N-dimensional histogram for weighted data with any axis types.");

    // Requieres sampled fills
    // register_histogram_by_type<axes::any, bh::profile_storage>(hist,
    //    "any_profile",
    //    "N-dimensional histogram for sampled data with any axis types.");

    // register_histogram_by_type<axes::any, bh::weighted_profile_storage>(hist,
    //    "any_weighted_profile",
    //    "N-dimensional histogram for weighted and sampled data with any axis types.");
}
