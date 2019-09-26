// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <boost/histogram/python/pybind11.hpp>

#include <boost/histogram/axis.hpp>

#include <algorithm>
#include <string>
#include <tuple>
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

class metadata_t : public py::object {
    PYBIND11_OBJECT_DEFAULT(metadata_t, object, PyObject_Check);

    bool operator==(const metadata_t &other) const {
        return py::cast<bool>(this->attr("__eq__")(other));
    }

    bool operator!=(const metadata_t &other) const {
        return py::cast<bool>(this->attr("__ne__")(other));
    }
};

namespace detail {
template <class Iterable>
std::string::size_type max_string_length(const Iterable &c) {
    std::string::size_type n = 0;
    for(auto &&s : c)
        n = std::max(n, s.size());
    return n;
}
} // namespace detail

namespace axis {

template <class A>
py::array bins_impl(const A &ax, bool flow) {
    const bh::axis::index_type underflow
        = flow && (bh::axis::traits::options(ax) & bh::axis::option::underflow);
    const bh::axis::index_type overflow
        = flow && (bh::axis::traits::options(ax) & bh::axis::option::overflow);

    py::array_t<double> result(
        {static_cast<std::size_t>(ax.size() + underflow + overflow), std::size_t(2)});

    for(auto i = -underflow; i < ax.size() + overflow; i++) {
        result.mutable_at(static_cast<std::size_t>(i + underflow), 0) = ax.value(i);
        result.mutable_at(static_cast<std::size_t>(i + underflow), 1) = ax.value(i + 1);
    }

    return std::move(result);
}

template <class... Ts>
py::array bins_impl(const bh::axis::integer<int, Ts...> &ax, bool flow) {
    const bh::axis::index_type underflow
        = flow && (bh::axis::traits::options(ax) & bh::axis::option::underflow);
    const bh::axis::index_type overflow
        = flow && (bh::axis::traits::options(ax) & bh::axis::option::overflow);

    py::array_t<int> result(static_cast<std::size_t>(ax.size() + underflow + overflow));

    for(auto i = -underflow; i < ax.size() + overflow; i++)
        result.mutable_at(static_cast<std::size_t>(i)) = ax.value(i);

    return std::move(result);
}

template <class... Ts>
py::array bins_impl(const bh::axis::category<int, Ts...> &ax, bool flow) {
    static_assert(
        !(bh::axis::category<int, Ts...>::options() & bh::axis::option::underflow),
        "category axis never has underflow");

    const bh::axis::index_type overflow
        = flow && (bh::axis::traits::options(ax) & bh::axis::option::overflow);

    py::array_t<int> result(static_cast<std::size_t>(ax.size() + overflow));

    for(auto i = 0; i < ax.size() + overflow; i++)
        result.mutable_at(static_cast<std::size_t>(i)) = ax.value(i);

    return std::move(result);
}

template <class... Ts>
py::array bins_impl(const bh::axis::category<std::string, Ts...> &ax, bool flow) {
    static_assert(!(bh::axis::category<std::string, Ts...>::options()
                    & bh::axis::option::underflow),
                  "category axis never has underflow");

    const bh::axis::index_type overflow
        = flow && (bh::axis::traits::options(ax) & bh::axis::option::overflow);

    const auto n = detail::max_string_length(ax);
    // TODO: this should return unicode
    py::array result(py::dtype(bh::detail::cat("S", n + 1)), ax.size() + overflow);

    for(auto i = 0; i < ax.size() + overflow; i++) {
        auto sout     = static_cast<char *>(result.mutable_data(i));
        const auto &s = ax.value(i);
        std::copy(s.begin(), s.end(), sout);
        sout[s.size()] = 0;
    }

    return result;
}

/// Utility to convert bins of axis to numpy array
template <class A>
py::array bins(const A &ax, bool flow) {
    // this indirection is needed by pybind11
    return bins_impl(ax, flow);
}

template <class A>
py::array edges_impl(const A &ax, bool flow) {
    const bh::axis::index_type underflow
        = flow && (bh::axis::traits::options(ax) & bh::axis::option::underflow);
    const bh::axis::index_type overflow
        = flow && (bh::axis::traits::options(ax) & bh::axis::option::overflow);

    py::array_t<double> edges(
        static_cast<std::size_t>(ax.size() + 1 + overflow + underflow));

    for(bh::axis::index_type i = -underflow; i <= ax.size() + overflow; ++i)
        edges.mutable_at(i + underflow) = ax.value(i);

    return std::move(edges);
}

template <class U, class... Ts>
py::array edges_impl(const bh::axis::category<U, Ts...> &ax, bool flow) {
    return bins(ax, flow);
}

/// Convert continuous axis into numpy.histogram compatible edge array
template <class A>
py::array edges(const A &ax, bool flow) {
    return edges_impl(ax, flow);
}

template <class A>
py::array centers(const A &ax) {
    return bh::detail::static_if<bh::axis::traits::is_continuous<A>>(
        [](const auto &ax) -> py::array {
            py::array_t<double> result(static_cast<std::size_t>(ax.size()));
            std::transform(ax.begin(),
                           ax.end(),
                           result.mutable_data(),
                           [](const auto &b) { return b.center(); });
            return std::move(result);
        },
        [](const auto &ax) { return bins(ax, false); },
        ax);
}

template <class A>
py::array_t<double> widths_impl(const A &ax) {
    py::array_t<double> result(static_cast<std::size_t>(ax.size()));
    std::transform(ax.begin(), ax.end(), result.mutable_data(), [](const auto &b) {
        return b.width();
    });
    return result;
}

template <class... Ts>
py::array_t<double> widths_impl(const bh::axis::integer<int, Ts...> &ax) {
    py::array_t<double> result(static_cast<std::size_t>(ax.size()));
    std::fill(result.mutable_data(), result.mutable_data() + ax.size(), 1.0);
    return result;
}

template <class U, class... Ts>
py::array_t<double> widths_impl(const bh::axis::category<U, Ts...> &ax) {
    py::array_t<double> result(static_cast<std::size_t>(ax.size()));
    std::fill(result.mutable_data(), result.mutable_data() + ax.size(), 1.0);
    return result;
}

template <class A>
py::array_t<double> widths(const A &ax) {
    return widths_impl(ax);
}

template <class A>
py::object unchecked_center(const A &ax, bh::axis::index_type i) {
    return bh::detail::static_if<bh::axis::traits::is_continuous<A>>(
        [i](const auto &ax) { return py::cast(ax.bin(i).center()); },
        [i](const auto &ax) { return py::cast(ax.bin(i)); },
        ax);
}

template <class A>
py::object unchecked_bin(const A &ax, bh::axis::index_type i) {
    return bh::detail::static_if<bh::axis::traits::is_continuous<A>>(
        [i](const auto &ax) -> py::object {
            return std::move(py::make_tuple(ax.bin(i).lower(), ax.bin(i).upper()));
        },
        [i](const auto &ax) -> py::object { return py::cast(ax.bin(i)); },
        ax);
}

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
