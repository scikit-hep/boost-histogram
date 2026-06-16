// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <bh_python/pybind11.hpp>

#include <bh_python/accumulators/mean.hpp>
#include <bh_python/accumulators/weighted_collector.hpp>
#include <bh_python/accumulators/weighted_mean.hpp>
#include <bh_python/accumulators/weighted_sum.hpp>
#include <bh_python/multi_cell.hpp>

#include <boost/histogram/accumulators/collector.hpp>
#include <boost/histogram/accumulators/count.hpp>
#include <boost/histogram/storage_adaptor.hpp>
#include <boost/histogram/unlimited_storage.hpp>

#include <vector>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace storage {

// Names match Python names
using int64         = bh::dense_storage<int64_t>;
using atomic_int64  = bh::dense_storage<bh::accumulators::count<int64_t, true>>;
using double_       = bh::dense_storage<double>;
using unlimited     = bh::unlimited_storage<>;
using weight        = bh::dense_storage<accumulators::weighted_sum<double>>;
using multi_cell    = bh::multi_cell<double>;
using mean          = bh::dense_storage<accumulators::mean<double>>;
using weighted_mean = bh::dense_storage<accumulators::weighted_mean<double>>;
using collector = bh::dense_storage<bh::accumulators::collector<std::vector<double>>>;
using weighted_collector = bh::dense_storage<accumulators::weighted_collector<double>>;

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
inline const char* name<multi_cell>() {
    return "multi_cell";
}

template <>
inline const char* name<mean>() {
    return "mean";
}

template <>
inline const char* name<weighted_mean>() {
    return "weighted_mean";
}

template <>
inline const char* name<collector>() {
    return "collector";
}

template <>
inline const char* name<weighted_collector>() {
    return "weighted_collector";
}

} // namespace storage

// It is very important that storages with accumulators have specialized serialization.
// No specializations needed for dense_storage<ArithmeticType> or unlimited_storage.

template <class Archive>
void save(Archive& ar, const storage::atomic_int64& s, unsigned /* version */) {
    // We cannot view the memory as a numpy array, because the internal layout of
    // std::atomic is undefined. So no reinterpret_casts are allowed.
    py::array_t<std::int64_t> a(static_cast<py::ssize_t>(s.size()));

    auto in_ptr   = s.begin();
    auto* out_ptr = a.mutable_data();
    for(; in_ptr != s.end(); ++in_ptr, ++out_ptr)
        *out_ptr = in_ptr->value();
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
    py::array_t<double> const a(static_cast<py::ssize_t>(s.size()) * 2,
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
    if(a.size() % 2 != 0)
        throw std::runtime_error(
            "weighted_sum: serialized buffer size is not a multiple of 2");
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
    py::array_t<double> const a(static_cast<py::ssize_t>(s.size()) * 3,
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
    if(a.size() % 3 != 0)
        throw std::runtime_error("mean: serialized buffer size is not a multiple of 3");
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
    py::array_t<double> const a(static_cast<py::ssize_t>(s.size()) * 4,
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
    if(a.size() % 4 != 0)
        throw std::runtime_error(
            "weighted_mean: serialized buffer size is not a multiple of 4");
    s.resize(static_cast<std::size_t>(a.size() / 4));
    // sadly we cannot move the memory from the numpy array into the vector
    std::copy(a.data(), a.data() + a.size(), reinterpret_cast<double*>(s.data()));
}

// The collector storage holds a variable-length std::vector<double> per bin, so it
// cannot be viewed as a flat numpy array like the other accumulators. We serialize it
// as a numpy array of per-bin counts plus a single numpy array of all values
// concatenated (the same offsets+content layout a future Awkward conversion would use).
template <class Archive>
void save(Archive& ar, const storage::collector& s, unsigned /* version */) {
    const auto n = s.size();
    py::array_t<std::int64_t> counts(static_cast<py::ssize_t>(n));
    auto* count_ptr   = counts.mutable_data();
    std::size_t total = 0;
    for(std::size_t i = 0; i < n; ++i) {
        const auto c = s[i].size();
        count_ptr[i] = static_cast<std::int64_t>(c);
        total += c;
    }
    py::array_t<double> values(static_cast<py::ssize_t>(total));
    auto* value_ptr = values.mutable_data();
    for(std::size_t i = 0; i < n; ++i) {
        const auto& cell = s[i];
        std::copy(cell.data(), cell.data() + cell.size(), value_ptr);
        value_ptr += cell.size();
    }
    ar << counts;
    ar << values;
}

template <class Archive>
void load(Archive& ar, storage::collector& s, unsigned /* version */) {
    py::array_t<std::int64_t> counts;
    py::array_t<double> values;
    ar >> counts;
    ar >> values;
    const auto n = static_cast<std::size_t>(counts.size());
    s.resize(n);
    const auto* count_ptr = counts.data();
    const auto* value_ptr = values.data();
    using collector_t     = bh::accumulators::collector<std::vector<double>>;
    for(std::size_t i = 0; i < n; ++i) {
        const auto c = static_cast<std::size_t>(count_ptr[i]);
        s[i]         = collector_t(value_ptr, value_ptr + c);
        value_ptr += c;
    }
}

// Same offsets+content layout as the collector above, with the per-bin
// (value, weight) pairs split into two flat numpy arrays.
template <class Archive>
void save(Archive& ar, const storage::weighted_collector& s, unsigned /* version */) {
    const auto n = s.size();
    py::array_t<std::int64_t> counts(static_cast<py::ssize_t>(n));
    auto* count_ptr   = counts.mutable_data();
    std::size_t total = 0;
    for(std::size_t i = 0; i < n; ++i) {
        const auto c = s[i].size();
        count_ptr[i] = static_cast<std::int64_t>(c);
        total += c;
    }
    py::array_t<double> values(static_cast<py::ssize_t>(total));
    py::array_t<double> weights(static_cast<py::ssize_t>(total));
    auto* value_ptr  = values.mutable_data();
    auto* weight_ptr = weights.mutable_data();
    for(std::size_t i = 0; i < n; ++i) {
        for(const auto& e : s[i]) {
            *value_ptr++  = e.value;
            *weight_ptr++ = e.weight;
        }
    }
    ar << counts;
    ar << values;
    ar << weights;
}

template <class Archive>
void load(Archive& ar, storage::weighted_collector& s, unsigned /* version */) {
    py::array_t<std::int64_t> counts;
    py::array_t<double> values;
    py::array_t<double> weights;
    ar >> counts;
    ar >> values;
    ar >> weights;
    const auto n = static_cast<std::size_t>(counts.size());
    s.resize(n);
    const auto* count_ptr  = counts.data();
    const auto* value_ptr  = values.data();
    const auto* weight_ptr = weights.data();
    using collector_t      = accumulators::weighted_collector<double>;
    for(std::size_t i = 0; i < n; ++i) {
        const auto c = static_cast<std::size_t>(count_ptr[i]);
        typename collector_t::container_type cont;
        cont.reserve(c);
        for(std::size_t j = 0; j < c; ++j)
            cont.emplace_back(value_ptr[j], weight_ptr[j]);
        value_ptr += c;
        weight_ptr += c;
        s[i] = collector_t(std::move(cont));
    }
}

namespace pybind11 {
namespace detail {
/// Allow a Python int to implicitly convert to an atomic int in C++
template <>
struct type_caster<storage::atomic_int64::value_type> {
    PYBIND11_TYPE_CASTER(storage::atomic_int64::value_type, _("atomic_int64"));

    bool load(handle src, bool) {
        auto* ptr = PyNumber_Long(src.ptr());
        if(ptr == nullptr)
            return false;
        value = PyLong_AsLongLong(ptr);
        Py_DECREF(ptr);
        return PyErr_Occurred() == nullptr;
    }

    static handle cast(const storage::atomic_int64::value_type& src,
                       return_value_policy /* policy */,
                       handle /* parent */) {
        return PyLong_FromLongLong(src.value());
    }
};
} // namespace detail
} // namespace pybind11
