// Copyright 2018-2019 Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <cstddef>

// minimal substitute for C++17 std::string_view
struct string_view {
    using value_type = char;

    string_view() = default;
    string_view(const char* d, std::size_t s)
        : data_(d)
        , size_(s) {}
    string_view(const string_view&) = default;
    string_view& operator=(const string_view&) = default;

    const char* data() const { return data_; }
    std::size_t size() const { return size_; }

    const char* data_ = nullptr;
    std::size_t size_ = 0;
};

namespace pybind11 {
namespace detail {
/// Allow conversion PyObject <-> our string_view
template <>
struct type_caster<string_view> : string_caster<string_view, true> {};
} // namespace detail
} // namespace pybind11
