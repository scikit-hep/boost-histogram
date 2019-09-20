// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#include <boost/histogram/python/pybind11.hpp>

#include <boost/histogram/python/axis.hpp>
#include <boost/histogram/python/histogram.hpp>
#include <boost/histogram/python/register_histogram.hpp>
#include <boost/histogram/python/storage.hpp>
#include <boost/histogram/storage_adaptor.hpp>

void register_general_histograms(py::module &hist) {
    register_histogram<vector_axis_variant, storage::int_>(
        hist,
        "_any_int",
        "N-dimensional histogram for unlimited size data with any axis types.");

    register_histogram<vector_axis_variant, storage::unlimited>(
        hist,
        "_any_unlimited",
        "N-dimensional histogram for unlimited size data with any axis types.");

    register_histogram<vector_axis_variant, storage::double_>(
        hist,
        "_any_double",
        "N-dimensional histogram for real-valued data with weights with any axis "
        "types.");

    register_histogram<vector_axis_variant, storage::atomic_int>(
        hist,
        "_any_atomic_int",
        "N-dimensional histogram for threadsafe integer data with any axis types.");

    register_histogram<vector_axis_variant, storage::weight>(
        hist,
        "_any_weight",
        "N-dimensional histogram for weighted data with any axis types.");

    register_histogram<vector_axis_variant, bh::profile_storage>(
        hist,
        "_any_profile",
        "N-dimensional histogram for sampled data with any axis types.");

    register_histogram<vector_axis_variant, bh::weighted_profile_storage>(
        hist,
        "_any_weighted_profile",
        "N-dimensional histogram for weighted and sampled data with any axis types.");
}
