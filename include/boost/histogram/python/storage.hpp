// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <boost/histogram/python/pybind11.hpp>

#include <boost/histogram.hpp>
#include <cstdint>
#include <atomic>


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

namespace storage {

// Names match Python names
using int_ = bh::dense_storage<uint64_t>;
using atomic_int = bh::dense_storage<copyable_atomic<uint64_t>>;
using double_ = bh::dense_storage<double>;
using unlimited = bh::unlimited_storage<>;
using weight = bh::weight_storage;
using profile = bh::profile_storage;
using weighted_profile = bh::weighted_profile_storage;
    
// Some types not yet suppored (mostly due to fill not accepting weight and sample yet)
using any_variant = bh::axis::variant<
    atomic_int,
    int_,
    double_,
    unlimited,
    weight
>;

}  // namespace storage
    
storage::any_variant extract_storage(py::kwargs kwargs);
