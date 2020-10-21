// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <bh_python/pybind11.hpp>

#include <bh_python/accumulators/mean.hpp>
#include <bh_python/accumulators/weighted_mean.hpp>
#include <bh_python/accumulators/weighted_sum.hpp>

#include <boost/histogram/accumulators/thread_safe.hpp>
#include <boost/histogram/storage_adaptor.hpp>
#include <boost/histogram/unlimited_storage.hpp>

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace storage {

// Names match Python names
using int64         = bh::dense_storage<uint64_t>;
using atomic_int64  = bh::dense_storage<bh::accumulators::thread_safe<uint64_t>>;
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
inline const char* name<int64>() {
    return "int64";
}

template <>
inline const char* name<atomic_int64>() {
    return "atomic_int64";
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

// It is very important that storages with accumulators have specialized serialization.
// No specializations needed for dense_storage<ArithmeticType> or unlimited_storage.

template <class Archive>
void save(Archive& ar, const storage::atomic_int64& s, unsigned /* version */) {
    // We cannot view the memory as a numpy array, because the internal layout of
    // std::atomic is undefined. So no reinterpret_casts are allowed.
    py::array_t<std::int64_t> a(static_cast<py::ssize_t>(s.size()));
    std::copy(s.begin(), s.end(), a.mutable_data());
    ar << a;
}

template <class Archive>
void load(Archive& ar, storage::atomic_int64& s, unsigned /* version */) {
    // data is stored as flat numpy array
    py::array_t<std::int64_t> a;
    ar >> a;
    s.resize(static_cast<std::size_t>(a.size()));
    // sadly we cannot move the memory from the numpy array into the vector
    std::copy(a.data(), a.data() + a.size(), s.data());
}

template <class Archive>
void save(Archive& ar,
          const bh::dense_storage<accumulators::weighted_sum<double>>& s,
          unsigned /* version */) {
    using T = accumulators::weighted_sum<double>;
    static_assert(std::is_standard_layout<T>::value
                      && std::is_trivially_copyable<T>::value
                      && sizeof(T) == 2 * sizeof(double),
                  "weighted_sum cannot be fast serialized");
    // view storage buffer as flat numpy array
    py::array_t<double> a(static_cast<py::ssize_t>(s.size()) * 2,
                          reinterpret_cast<const double*>(s.data()));
    ar << a;
}

template <class Archive>
void load(Archive& ar,
          bh::dense_storage<accumulators::weighted_sum<double>>& s,
          unsigned /* version */) {
    // data is stored as flat numpy array
    py::array_t<double> a;
    ar >> a;
    s.resize(static_cast<std::size_t>(a.size() / 2));
    // sadly we cannot move the memory from the numpy array into the vector
    std::copy(a.data(), a.data() + a.size(), reinterpret_cast<double*>(s.data()));
}

template <class Archive>
void save(Archive& ar,
          const bh::dense_storage<accumulators::mean<double>>& s,
          unsigned /* version */) {
    using T = accumulators::mean<double>;
    static_assert(std::is_standard_layout<T>::value
                      && std::is_trivially_copyable<T>::value
                      && sizeof(T) == 3 * sizeof(double),
                  "mean cannot be fast serialized");
    // view storage buffer as flat numpy array
    py::array_t<double> a(static_cast<py::ssize_t>(s.size()) * 3,
                          reinterpret_cast<const double*>(s.data()));
    ar << a;
}

template <class Archive>
void load(Archive& ar,
          bh::dense_storage<accumulators::mean<double>>& s,
          unsigned /* version */) {
    // data is stored as flat numpy array
    py::array_t<double> a;
    ar >> a;
    s.resize(static_cast<std::size_t>(a.size() / 3));
    // sadly we cannot move the memory from the numpy array into the vector
    std::copy(a.data(), a.data() + a.size(), reinterpret_cast<double*>(s.data()));
}

template <class Archive>
void save(Archive& ar,
          const bh::dense_storage<accumulators::weighted_mean<double>>& s,
          unsigned /* version */) {
    using T = accumulators::weighted_mean<double>;
    static_assert(std::is_standard_layout<T>::value
                      && std::is_trivially_copyable<T>::value
                      && sizeof(T) == 4 * sizeof(double),
                  "weighted_mean cannot be fast serialized");
    // view storage buffer as flat numpy array
    py::array_t<double> a(static_cast<py::ssize_t>(s.size()) * 4,
                          reinterpret_cast<const double*>(s.data()));
    ar << a;
}

template <class Archive>
void load(Archive& ar,
          bh::dense_storage<accumulators::weighted_mean<double>>& s,
          unsigned /* version */) {
    // data is stored as flat numpy array
    py::array_t<double> a;
    ar >> a;
    s.resize(static_cast<std::size_t>(a.size() / 4));
    // sadly we cannot move the memory from the numpy array into the vector
    std::copy(a.data(), a.data() + a.size(), reinterpret_cast<double*>(s.data()));
}

namespace pybind11 {
namespace detail {
/// Allow a Python int to implicitly convert to an atomic int in C++
template <>
struct type_caster<storage::atomic_int64::value_type> {
    PYBIND11_TYPE_CASTER(storage::atomic_int64::value_type, _("atomic_int64"));

    bool load(handle src, bool) {
        auto ptr = PyNumber_Long(src.ptr());
        if(!ptr)
            return false;
        value.store(PyLong_AsUnsignedLongLong(ptr));
        Py_DECREF(ptr);
        return !PyErr_Occurred();
    }

    static handle cast(storage::atomic_int64::value_type src,
                       return_value_policy /* policy */,
                       handle /* parent */) {
        return PyLong_FromUnsignedLongLong(src.load());
    }
};
} // namespace detail
} // namespace pybind11
