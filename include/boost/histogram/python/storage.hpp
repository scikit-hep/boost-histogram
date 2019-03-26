// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <boost/histogram/python/pybind11.hpp>

#include <boost/histogram.hpp>
#include <cstdint>
#include <atomic>

namespace bh = boost::histogram;

/*
 std::atomic has deleted copy ctor, we need to wrap it in a type with a
 potentially unsafe copy ctor. It can be used in a thread-safe way if
 it is not in a growing axis type.
 */
template <typename T>
class copyable_atomic : public std::atomic<T> {
public:
    using std::atomic<T>::atomic;
    
    // zero-initialize the atomic T
    copyable_atomic() noexcept : std::atomic<T>(T()) {}
    
    // this is potentially not thread-safe, see below
    copyable_atomic(const copyable_atomic& rhs) : std::atomic<T>() { this->operator=(rhs); }
    
    // this is potentially not thread-safe, see below
    copyable_atomic& operator=(const copyable_atomic& rhs) {
        if (this != &rhs) { std::atomic<T>::store(rhs.load()); }
        return *this;
    }
};


using dense_int_storage = bh::dense_storage<uint64_t>;
using dense_atomic_int_storage = bh::dense_storage<copyable_atomic<uint64_t>>;
using dense_double_storage = bh::dense_storage<double>;

using any_storage_variant = bh::axis::variant<
    dense_atomic_int_storage,
    dense_int_storage,
    dense_double_storage,
    bh::unlimited_storage<>,
    bh::weight_storage
>;

any_storage_variant extract_storage(py::kwargs kwargs);
