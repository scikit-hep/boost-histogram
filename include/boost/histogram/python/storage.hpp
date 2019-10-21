// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <boost/histogram/python/pybind11.hpp>

#include <boost/histogram/accumulators.hpp>
#include <boost/histogram/storage_adaptor.hpp>
#include <boost/histogram/unlimited_storage.hpp>

#include <cstdint>

namespace storage {

// Names match Python names
using int_          = bh::dense_storage<uint64_t>;
using atomic_int    = bh::dense_storage<bh::accumulators::thread_safe<uint64_t>>;
using double_       = bh::dense_storage<double>;
using unlimited     = bh::unlimited_storage<>;
using weight        = bh::weight_storage;
using mean          = bh::profile_storage;
using weighted_mean = bh::weighted_profile_storage;

// Allow repr to show python name
template <class S>
inline const char *name() {
    return "unknown";
}

template <>
inline const char *name<int_>() {
    return "int";
}

template <>
inline const char *name<atomic_int>() {
    return "atomic_int";
}

template <>
inline const char *name<double_>() {
    return "double";
}

template <>
inline const char *name<unlimited>() {
    return "unlimited";
}

template <>
inline const char *name<weight>() {
    return "weight";
}

template <>
inline const char *name<mean>() {
    return "mean";
}

template <>
inline const char *name<weighted_mean>() {
    return "weighted_mean";
}

} // namespace storage
