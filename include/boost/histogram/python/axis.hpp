// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <boost/histogram/python/pybind11.hpp>

#include <boost/histogram/axis.hpp>
#include <boost/histogram/python/typetools.hpp>

#include <tuple>
#include <vector>

/// Register bh::axis::variant as a variant for PyBind11
namespace pybind11 {
namespace detail {
template <class... Ts>
struct type_caster<bh::axis::variant<Ts...>>
    : variant_caster<bh::axis::variant<Ts...>> {};
} // namespace detail
} // namespace pybind11

/// Utility to convert an axis to edges array
template <class A>
py::array_t<double> axis_to_edges(const A &ax, bool flow) {
    unsigned overflow
        = flow && (bh::axis::traits::options(ax) & bh::axis::option::underflow);
    unsigned underflow
        = flow && (bh::axis::traits::options(ax) & bh::axis::option::overflow);

    py::array_t<double> edges((unsigned)ax.size() + 1u + overflow + underflow);

    if(underflow)
        edges.mutable_at(0u) = ax.bin(-1).lower();

    edges.mutable_at(0u + underflow) = ax.bin(0).lower();

    std::transform(ax.begin(),
                   ax.end(),
                   edges.mutable_data() + 1u + underflow,
                   [](const auto &bin) { return bin.upper(); });

    if(overflow)
        edges.mutable_at(edges.size() - 1) = ax.bin(ax.size()).upper();

    return edges;
}

template <class A>
decltype(auto) axis_to_bins(const A &self, bool flow) {
    std::vector<bh::python::remove_cvref_t<decltype(self.bin(std::declval<int>()))>>
        out;
    bool overflow
        = flow && (bh::axis::traits::options(self) & bh::axis::option::underflow);
    bool underflow
        = flow && (bh::axis::traits::options(self) & bh::axis::option::overflow);

    out.reserve((size_t)bh::axis::traits::extent(self));

    for(int i = 0 - underflow; i < self.size() + overflow; i++)
        out.emplace_back(self.bin(i));

    return out;
}

inline bool PyObject_Check(void *value) { return value != nullptr; }

class metadata_t : public py::object {
    PYBIND11_OBJECT_DEFAULT(metadata_t, object, PyObject_Check);

    bool operator==(const metadata_t &other) const {
        return py::cast<bool>(this->attr("__eq__")(other));
    }

    bool operator!=(const metadata_t &other) const {
        return py::cast<bool>(this->attr("__ne__")(other));
    }
};

namespace axis {

// These match the Python names

using _regular_uoflow = bh::axis::regular<double, bh::use_default, metadata_t>;
using _regular_uflow  = bh::axis::
    regular<double, bh::use_default, metadata_t, bh::axis::option::underflow_t>;
using _regular_oflow = bh::axis::
    regular<double, bh::use_default, metadata_t, bh::axis::option::overflow_t>;
using _regular_noflow
    = bh::axis::regular<double, bh::use_default, metadata_t, bh::axis::option::none_t>;
using _regular_growth = bh::axis::
    regular<double, bh::use_default, metadata_t, bh::axis::option::growth_t>;

using circular     = bh::axis::circular<double, metadata_t>;
using regular_log  = bh::axis::regular<double, bh::axis::transform::log, metadata_t>;
using regular_sqrt = bh::axis::regular<double, bh::axis::transform::sqrt, metadata_t>;
using regular_pow  = bh::axis::regular<double, bh::axis::transform::pow, metadata_t>;

using _variable_uoflow = bh::axis::variable<double, metadata_t>;
using _variable_uflow
    = bh::axis::variable<double, metadata_t, bh::axis::option::underflow_t>;
using _variable_oflow
    = bh::axis::variable<double, metadata_t, bh::axis::option::overflow_t>;
using _variable_noflow
    = bh::axis::variable<double, metadata_t, bh::axis::option::none_t>;

using _integer_uoflow = bh::axis::integer<int, metadata_t>;
using _integer_uflow
    = bh::axis::integer<int, metadata_t, bh::axis::option::underflow_t>;
using _integer_oflow = bh::axis::integer<int, metadata_t, bh::axis::option::overflow_t>;
using _integer_noflow = bh::axis::integer<int, metadata_t, bh::axis::option::none_t>;
using _integer_growth = bh::axis::integer<int, metadata_t, bh::axis::option::growth_t>;

using _category_int = bh::axis::category<int, metadata_t>;
using _category_int_growth
    = bh::axis::category<int, metadata_t, bh::axis::option::growth_t>;

using _category_str = bh::axis::category<std::string, metadata_t>;
using _category_str_growth
    = bh::axis::category<std::string, metadata_t, bh::axis::option::growth_t>;

} // namespace axis

// The following list is all types supported
using axis_variant = bh::axis::variant<axis::_regular_uoflow,
                                       axis::_regular_uflow,
                                       axis::_regular_oflow,
                                       axis::_regular_noflow,
                                       axis::_regular_growth,
                                       axis::circular,
                                       axis::regular_log,
                                       axis::regular_pow,
                                       axis::regular_sqrt,
                                       axis::_variable_uoflow,
                                       axis::_variable_oflow,
                                       axis::_variable_uflow,
                                       axis::_variable_noflow,
                                       axis::_integer_uoflow,
                                       axis::_integer_oflow,
                                       axis::_integer_uflow,
                                       axis::_integer_noflow,
                                       axis::_integer_growth,
                                       axis::_category_int,
                                       axis::_category_int_growth,
                                       axis::_category_str,
                                       axis::_category_str_growth>;

// This saves a little typing
using vector_axis_variant = std::vector<axis_variant>;
