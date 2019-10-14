// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <boost/histogram/python/pybind11.hpp>

#include <boost/histogram/axis.hpp>

#include <string>
#include <type_traits>
#include <vector>

/// Register bh::axis::variant as a variant for PyBind11
namespace pybind11 {
namespace detail {
template <class... Ts>
struct type_caster<bh::axis::variant<Ts...>>
    : variant_caster<bh::axis::variant<Ts...>> {};
} // namespace detail
} // namespace pybind11

inline bool PyObject_Check(void *value) { return value != nullptr; }

struct metadata_t : py::object {
    PYBIND11_OBJECT_DEFAULT(metadata_t, object, PyObject_Check);

    bool operator==(const metadata_t &other) const { return py::object::equal(other); }
    bool operator!=(const metadata_t &other) const {
        return py::object::not_equal(other);
    }
};

template <class Iterable>
std::string::size_type max_string_length(const Iterable &c) {
    std::string::size_type n = 0;
    for(auto &&s : c)
        n = std::max(n, s.size());
    return n;
}
