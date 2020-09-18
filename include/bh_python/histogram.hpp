// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <bh_python/pybind11.hpp>

#include <bh_python/accumulators/mean.hpp>
#include <bh_python/accumulators/weighted_mean.hpp>
#include <bh_python/accumulators/weighted_sum.hpp>

#include <boost/histogram/detail/axes.hpp>
#include <boost/histogram/histogram.hpp>
#include <boost/histogram/unsafe_access.hpp>

namespace pybind11 {

/// The descriptor for atomic_* is the same as the descriptor for *, as long this uses
/// standard layout
template <class T>
struct format_descriptor<bh::accumulators::thread_safe<T>> : format_descriptor<T> {
    static_assert(std::is_standard_layout<bh::accumulators::thread_safe<T>>::value, "");
};

} // namespace pybind11

namespace detail {

template <class Axes, class T>
py::buffer_info make_buffer_impl(const Axes& axes, bool flow, T* ptr) {
    // strides are in bytes
    auto shape     = bh::detail::make_stack_buffer<ssize_t>(axes);
    auto strides   = bh::detail::make_stack_buffer<ssize_t>(axes);
    ssize_t stride = sizeof(T);
    unsigned rank  = 0;
    char* start    = reinterpret_cast<char*>(ptr);
    bh::detail::for_each_axis(axes, [&](const auto& axis) {
        const bool underflow
            = bh::axis::traits::options(axis) & bh::axis::option::underflow;
        if(!flow && underflow)
            start += stride;
        const auto extent = bh::axis::traits::extent(axis);
        shape[rank]       = flow ? extent : axis.size();
        strides[rank]     = stride;
        stride *= extent;
        ++rank;
    });

    return py::buffer_info(
        start,                              // Pointer to buffer
        sizeof(T),                          // Size of one scalar
        py::format_descriptor<T>::format(), // Python format descriptor
        rank,                               // Number of dimensions
        shape,                              // Buffer shape
        strides                             // Strides (in bytes) for each index
    );
}

struct double_converter {
    template <class T, class Buffer>
    void operator()(T* tp, Buffer& b) const {
        b.template make<double>(b.size, tp);
    }

    template <class Buffer>
    void operator()(double*, Buffer&) const {} // nothing to do
};
} // namespace detail

/// Build and return a buffer over the current data.
/// Flow controls whether under/over flow bins are present
template <class A, class T>
py::buffer_info make_buffer(bh::histogram<A, bh::dense_storage<T>>& h, bool flow) {
    const auto& axes = bh::unsafe_access::axes(h);
    auto& storage    = bh::unsafe_access::storage(h);
    return detail::make_buffer_impl(axes, flow, &storage[0]);
}

/// Specialization for unlimited_buffer
template <class A, class Allocator>
py::buffer_info make_buffer(bh::histogram<A, bh::unlimited_storage<Allocator>>& h,
                            bool flow) {
    const auto& axes = bh::unsafe_access::axes(h);
    auto& storage    = bh::unsafe_access::storage(h);
    // User requested a view into the memory of unlimited storage. We convert
    // the internal storage to double now to avoid the view becoming invalid
    // upon changes to the histogram. This is the only way to provide a safe
    // view, because we cannot automatically update the view when the
    // underlying memory buffer changes. In practice it is ok, because users
    // usually want to get the view after filling the histogram, and then the
    // counts are usually converted to doubles for further processing anyway.
    auto& buffer = bh::unsafe_access::unlimited_storage_buffer(storage);
    buffer.visit(detail::double_converter(), buffer);
    return detail::make_buffer_impl(axes, flow, static_cast<double*>(buffer.ptr));
}

/// Compute the bin of an array from a runtime list
/// For example, [1,3,2] will return that bin of an array
template <class F, int Opt>
const F& pyarray_at(const py::array_t<F, Opt>& input,
                    std::vector<py::ssize_t> indexes) {
    const auto rank = static_cast<py::ssize_t>(indexes.size());
    if(input.ndim() != rank)
        throw py::value_error(
            "The input array dimensions must match the index dimensions");

    const auto* strides     = input.strides();
    py::ssize_t flat_offset = 0;

    for(py::ssize_t r = 0; r < rank; r++) {
        auto input_shape  = input.shape(r);
        py::ssize_t index = indexes[static_cast<std::size_t>(r)];
        if(index < 0 || index > input_shape)
            throw py::value_error("The index must not be outside the array boundaries");

        flat_offset += (strides[r] / static_cast<py::ssize_t>(sizeof(F))) * index;
    }
    return *(input.data() + flat_offset);
}
