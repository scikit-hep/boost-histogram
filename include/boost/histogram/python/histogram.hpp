// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <boost/histogram/python/pybind11.hpp>

#include <boost/histogram.hpp>

#include <vector>

namespace bh = boost::histogram;


template<typename T>
struct remove_atomic {
    using type = T;
};

template<>
struct remove_atomic<copyable_atomic<uint64_t>> {
    using type = std::uint64_t;
};


/// Build and return a buffer over the current data.
/// Could be optimized using array and maximum number of dims.
/// Flow controls whether under/over flow bins are present
template<typename A, typename S>
py::buffer_info make_buffer(bh::histogram<A, S>& h, bool flow) {
    using in_storage_t = typename bh::histogram<A, S>::value_type;
    using in_storage_value_t = typename remove_atomic<in_storage_t>::type;

    auto rank = h.rank();
    std::vector<ssize_t> diminsions, strides;
    ssize_t factor = 1;

    ssize_t start = 0;
    ssize_t size_of = sizeof(in_storage_t);

    for (unsigned i=0; i<rank; i++) {
        bool underflow = bh::axis::traits::options(h.axis(i)) & bh::axis::option::underflow;
        ssize_t extent_dim = bh::axis::traits::extent(h.axis(i));
        ssize_t size_dim = h.axis(i).size();
        if(!flow && underflow)
            start += factor;
        diminsions.push_back(flow ? extent_dim : size_dim);
        strides.push_back(factor * size_of);
        factor *= extent_dim;
    }

    return py::buffer_info(
                           &(*h.begin()) + start,                         // Pointer to buffer
                           sizeof(in_storage_t),                          // Size of one scalar
                           py::format_descriptor<in_storage_value_t>::format(), // Python struct-style format descriptor
                           rank,                                          // Number of dimensions
                           diminsions,                                    // Buffer dimensions
                           strides                                        // Strides (in bytes) for each index
                           );
}

/// Unlimited storage does not support buffer access
template<typename A>
py::buffer_info make_buffer(bh::histogram<A, bh::default_storage>&, bool) {
    return py::buffer_info();
}
