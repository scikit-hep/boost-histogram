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

template <>
inline void copy_in<>(bh::histogram<vector_axis_variant, bh::dense_storage<double>> &h,
                      const py::array_t<double> &input) {
    // Works on simple datatypes only
    // TODO: Add other types
    if(h.rank() != input.ndim())
        throw py::value_error(
            "The input array dimensions must match the histogram rank");

    // Quick check to ensure the input array is valid
    for(unsigned r = 0; r < h.rank(); r++) {
        auto input_shape = input.shape(static_cast<py::ssize_t>(r));
        if(input_shape != bh::axis::traits::extent(h.axis(r))
           && input_shape != h.axis(r).size() && input_shape != 1)
            throw py::value_error("The input array sizes must match the histogram "
                                  "(with or without flow), or be broadcastable to it");
    }

    std::vector<py::ssize_t> indexes;
    indexes.resize(h.rank());

    for(auto &&ind : bh::indexed(h, bh::coverage::all)) {
        bool skip = false;

        for(unsigned r = 0; r < h.rank(); r++) {
            auto input_shape   = input.shape(static_cast<py::ssize_t>(r));
            bool use_flow      = input_shape == bh::axis::traits::extent(h.axis(r));
            bool has_underflow = h.axis(r).options() & bh::axis::option::underflow;

            // Broadcast size 1
            if(input_shape == 1)
                indexes[r] = 0;

            // If this is using flow bins and has an underflow bin, convert -1 to 0
            // (etc)
            else if(use_flow && has_underflow)
                indexes[r] = ind.index(r) + 1;

            // If not using flow bins, skip the flow bins
            else if(!use_flow
                    && (ind.index(r) < 0 || ind.index(r) >= h.axis(r).size())) {
                skip = true;
                break;

                // Otherwise, this is normal
            } else
                indexes[r] = ind.index(r);
        }

        if(skip)
            continue;

        *ind = pyarray_at(input, indexes);
    }
}

void register_histograms(py::module &hist) {
    hist.attr("_axes_limit") = BOOST_HISTOGRAM_DETAIL_AXES_LIMIT;

    register_histogram<vector_axis_variant, storage::int_>(
        hist,
        "any_int",
        "N-dimensional histogram for unlimited size data with any axis types.");

    register_histogram<vector_axis_variant, storage::unlimited>(
        hist,
        "any_unlimited",
        "N-dimensional histogram for unlimited size data with any axis types.");

    register_histogram<vector_axis_variant, storage::double_>(
        hist,
        "any_double",
        "N-dimensional histogram for real-valued data with weights with any axis "
        "types.");

    register_histogram<vector_axis_variant, storage::atomic_int>(
        hist,
        "any_atomic_int",
        "N-dimensional histogram for threadsafe integer data with any axis types.");

    register_histogram<vector_axis_variant, storage::weight>(
        hist,
        "any_weight",
        "N-dimensional histogram for weighted data with any axis types.");

    register_histogram<vector_axis_variant, storage::mean>(
        hist,
        "any_mean",
        "N-dimensional histogram for sampled data with any axis types.");

    register_histogram<vector_axis_variant, storage::weighted_mean>(
        hist,
        "any_weighted_mean",
        "N-dimensional histogram for weighted and sampled data with any axis types.");
}
