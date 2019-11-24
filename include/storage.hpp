// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include "pybind11.hpp"

#include "accumulators/mean.hpp"
#include "accumulators/weighted_mean.hpp"
#include "accumulators/weighted_sum.hpp"

#include <boost/histogram/accumulators/sum.hpp>
#include <boost/histogram/accumulators/thread_safe.hpp>
#include <boost/histogram/storage_adaptor.hpp>
#include <boost/histogram/unlimited_storage.hpp>

#include <cstdint>

namespace storage {

// Names match Python names
using int_          = bh::dense_storage<uint64_t>;
using atomic_int    = bh::dense_storage<bh::accumulators::thread_safe<uint64_t>>;
using double_       = bh::dense_storage<double>;
using unlimited     = bh::unlimited_storage<>;
using weight        = bh::dense_storage<accumulators::weighted_sum<double>>;
using mean          = bh::dense_storage<accumulators::mean<double>>;
using weighted_mean = bh::dense_storage<accumulators::weighted_mean<double>>;

// Allow repr to show python name
template <class S>
inline const char* name() {
    return "unknown";
}

template <>
inline const char* name<int_>() {
    return "int";
}

template <>
inline const char* name<atomic_int>() {
    return "atomic_int";
}

template <>
inline const char* name<double_>() {
    return "double";
}

template <>
inline const char* name<unlimited>() {
    return "unlimited";
}

template <>
inline const char* name<weight>() {
    return "weight";
}

template <>
inline const char* name<mean>() {
    return "mean";
}

template <>
inline const char* name<weighted_mean>() {
    return "weighted_mean";
}

} // namespace storage

namespace pybind11 {
namespace detail {
/// Allow a Python int to implicitly convert to an atomic int in C++
template <>
struct type_caster<storage::atomic_int::value_type> {
    PYBIND11_TYPE_CASTER(storage::atomic_int::value_type, _("atomic_int"));

    bool load(handle src, bool) {
        auto ptr = PyNumber_Long(src.ptr());
        if(!ptr)
            return false;
        value.store(PyLong_AsUnsignedLongLong(ptr));
        Py_DECREF(ptr);
        return !PyErr_Occurred();
    }

    static handle cast(storage::atomic_int::value_type src,
                       return_value_policy /* policy */,
                       handle /* parent */) {
        return PyLong_FromUnsignedLongLong(src.load());
    }
};
} // namespace detail
} // namespace pybind11
