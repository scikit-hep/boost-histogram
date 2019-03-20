// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <boost/histogram/python/pybind11.hpp>

#include <boost/histogram.hpp>

#include <vector>

namespace bh = boost::histogram;

/// Build and return a buffer over the current data.
/// Could be optimized using array and maximum number of dims.
template<typename A, typename S>
py::buffer_info make_buffer(bh::histogram<A, S>& h) {
    using in_storage_t = typename bh::histogram<A, S>::value_type;
    
    auto rank = h.rank();
    std::vector<ssize_t> diminsions, strides;
    ssize_t factor = sizeof(in_storage_t);
    
    for (unsigned i=0; i<rank; i++) {
        auto dim = bh::axis::traits::extent(h.axis(i)); // (under/overflow)
        diminsions.push_back(dim);
        strides.push_back(factor);
        factor *= dim;
    }
    
    return py::buffer_info(
                           &(*h.begin()),                                 // Pointer to buffer
                           sizeof(in_storage_t),                          // Size of one scalar
                           py::format_descriptor<in_storage_t>::format(), // Python struct-style format descriptor
                           rank,                                          // Number of dimensions
                           diminsions,                                    // Buffer dimensions
                           strides                                        // Strides (in bytes) for each index
                           );
}

/// Unlimited storage does not support buffer access
template<typename A>
py::buffer_info make_buffer(bh::histogram<A, bh::default_storage>&) {
    return py::buffer_info();
}
