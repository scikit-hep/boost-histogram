// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <bh_python/pybind11.hpp>

#include <bh_python/regular_numpy.hpp>
#include <bh_python/transform.hpp>

#include <boost/histogram/axis.hpp>
#include <boost/histogram/indexed.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace axis {

namespace option = bh::axis::option;

using ogrowth_t  = decltype(option::growth | option::overflow);
using uogrowth_t = decltype(option::growth | option::underflow | option::overflow);
using circular_t = decltype(option::circular | option::overflow);

// Must be specialized for each type (compile warning if not)
template <class T>
inline const char* string_name();

// Macro to make the string specializations more readable
#define BHP_SPECIALIZE_NAME(name)                                                      \
    template <>                                                                        \
    inline const char* string_name<name>() {                                           \
        return #name;                                                                  \
    }

// These match the Python names
using regular_none
    = bh::axis::regular<double, bh::use_default, metadata_t, option::none_t>;
using regular_uflow
    = bh::axis::regular<double, bh::use_default, metadata_t, option::underflow_t>;
using regular_oflow
    = bh::axis::regular<double, bh::use_default, metadata_t, option::overflow_t>;
using regular_uoflow = bh::axis::regular<double, bh::use_default, metadata_t>;
using regular_uoflow_growth
    = bh::axis::regular<double, bh::use_default, metadata_t, uogrowth_t>;
using regular_circular
    = bh::axis::regular<double, bh::use_default, metadata_t, circular_t>;

BHP_SPECIALIZE_NAME(regular_none)
BHP_SPECIALIZE_NAME(regular_uflow)
BHP_SPECIALIZE_NAME(regular_oflow)
BHP_SPECIALIZE_NAME(regular_uoflow)
BHP_SPECIALIZE_NAME(regular_uoflow_growth)
BHP_SPECIALIZE_NAME(regular_circular)

using regular_pow   = bh::axis::regular<double, bh::axis::transform::pow, metadata_t>;
using regular_trans = bh::axis::regular<double, func_transform, metadata_t>;

BHP_SPECIALIZE_NAME(regular_pow)
BHP_SPECIALIZE_NAME(regular_trans)

using variable_none   = bh::axis::variable<double, metadata_t, option::none_t>;
using variable_uflow  = bh::axis::variable<double, metadata_t, option::underflow_t>;
using variable_oflow  = bh::axis::variable<double, metadata_t, option::overflow_t>;
using variable_uoflow = bh::axis::variable<double, metadata_t>;
using variable_uoflow_growth = bh::axis::variable<double, metadata_t, uogrowth_t>;
using variable_circular      = bh::axis::variable<double, metadata_t, circular_t>;

BHP_SPECIALIZE_NAME(variable_none)
BHP_SPECIALIZE_NAME(variable_uflow)
BHP_SPECIALIZE_NAME(variable_oflow)
BHP_SPECIALIZE_NAME(variable_uoflow)
BHP_SPECIALIZE_NAME(variable_uoflow_growth)
BHP_SPECIALIZE_NAME(variable_circular)

using integer_none     = bh::axis::integer<int, metadata_t, option::none_t>;
using integer_uoflow   = bh::axis::integer<int, metadata_t>;
using integer_uflow    = bh::axis::integer<int, metadata_t, option::underflow_t>;
using integer_oflow    = bh::axis::integer<int, metadata_t, option::overflow_t>;
using integer_growth   = bh::axis::integer<int, metadata_t, option::growth_t>;
using integer_circular = bh::axis::integer<int, metadata_t, option::circular_t>;

BHP_SPECIALIZE_NAME(integer_none)
BHP_SPECIALIZE_NAME(integer_uoflow)
BHP_SPECIALIZE_NAME(integer_uflow)
BHP_SPECIALIZE_NAME(integer_oflow)
BHP_SPECIALIZE_NAME(integer_growth)
BHP_SPECIALIZE_NAME(integer_circular)

using category_int        = bh::axis::category<int, metadata_t>;
using category_int_growth = bh::axis::category<int, metadata_t, option::growth_t>;

BHP_SPECIALIZE_NAME(category_int)
BHP_SPECIALIZE_NAME(category_int_growth)

using category_str = bh::axis::category<std::string, metadata_t, option::overflow_t>;
using category_str_growth
    = bh::axis::category<std::string, metadata_t, option::growth_t>;

BHP_SPECIALIZE_NAME(category_str)
BHP_SPECIALIZE_NAME(category_str_growth)

class boolean : public bh::axis::integer<int, metadata_t, option::none_t> {
  public:
    explicit boolean(metadata_t meta = {})
        : integer(0, 2, std::move(meta)) {}
    explicit boolean(const boolean& src,
                     bh::axis::index_type begin,
                     bh::axis::index_type end,
                     unsigned merge)
        : integer(src, begin, end, merge) {}
    boolean(const boolean& other) = default;

    bh::axis::index_type index(int x) const noexcept {
        return integer::index(x == 0 ? 0 : 1);
    }

    // We can't specify inclusive, since this could be sliced
};

// Built-in boolean requires bool fill, slower compile, not reducible
// using boolean = bh::axis::boolean<metadata_t>;
BHP_SPECIALIZE_NAME(boolean)

// Axis defined elsewhere
BHP_SPECIALIZE_NAME(regular_numpy)

#undef BHP_SPECIALIZE_NAME

// How edges, centers, and widths are handled
//
// We distinguish between continuous and discrete axes. The integer axis is a borderline
// case. It has discrete values, but they are consecutive. It is possible although not
// correct to treat it like a continuous axis with bin width of 1. For the sake of
// computing bin edges and bin center, we will use this ansatz here. PS: This behavior
// is already implemented in Boost::Histogram when you create an integer axis with a
// floating point value type, e.g. integer<double>. In this case, the integer axis acts
// strictly like a regular axis with a fixed bin with of 1. We don't use it here,
// because it is slightly slower than the integer<int> axis when the input values are
// truly integers.
//
// The category axis is treated like regular(size(), 0, size()) in the conversion. It is
// the responsibility of the user to set the labels accordingly when a histogram with a
// category axis is plotted.

template <class A>
constexpr bool is_category(const A&) {
    return false;
}

template <class... Ts>
constexpr bool is_category(const bh::axis::category<Ts...>&) {
    return true;
}

template <class Continuous, class Discrete, class Integer, class A>
decltype(auto) select(Continuous&& c, Discrete&& d, Integer&&, const A& ax) {
    return bh::detail::static_if<bh::axis::traits::is_continuous<A>>(
        std::forward<Continuous>(c), std::forward<Discrete>(d), ax);
}

template <class Continuous, class Discrete, class Integer, class... Ts>
decltype(auto)
select(Continuous&&, Discrete&&, Integer&& i, const bh::axis::integer<int, Ts...>& ax) {
    return std::forward<Integer>(i)(ax);
}

template <class Continuous, class Discrete, class Boolean, class... Ts>
decltype(auto) select(Continuous&&, Discrete&&, Boolean&& i, const boolean& ax) {
    return std::forward<Boolean>(i)(ax);
}

/// Return bin center for continuous axis and bin value for discrete axis
template <class A>
double unchecked_center(const A& ax, bh::axis::index_type i) {
    return select([i](const auto& ax) { return ax.value(i + 0.5); },
                  [i](const auto&) { return i + 0.5; },
                  [i](const auto& ax) { return ax.value(i) + 0.5; },
                  ax);
}

/// Return bin in a native Python representation
template <class A>
decltype(auto) unchecked_bin(const A& ax, bh::axis::index_type i) {
    return bh::detail::static_if<bh::axis::traits::is_continuous<A>>(
        [i](const auto& ax) -> decltype(auto) {
            return py::make_tuple(ax.value(i), ax.value(i + 1));
        },
        [i](const auto& ax) -> decltype(auto) {
            return (!is_category(ax) || i < ax.size()) ? py::cast(ax.bin(i))
                                                       : py::none();
        },
        ax);
}

/// Convert continuous axis into numpy.histogram compatible edge array
template <class A>
py::array_t<double> edges(const A& ax, bool flow = false, bool numpy_upper = false) {
    auto continuous = [flow, numpy_upper](const auto& ax) {
        using AX         = std::decay_t<decltype(ax)>;
        using index_type = bh::axis::index_type;

        const index_type underflow
            = flow && (bh::axis::traits::get_options<AX>::test(option::underflow));
        const index_type overflow
            = flow && (bh::axis::traits::get_options<AX>::test(option::overflow));

        py::array_t<double> edges(
            static_cast<py::ssize_t>(ax.size() + 1 + overflow + underflow));

        for(index_type i = -underflow; i <= ax.size() + overflow; ++i)
            edges.mutable_at(i + underflow) = ax.value(i);

        if(numpy_upper && !std::is_same<A, axis::regular_numpy>::value) {
            edges.mutable_at(ax.size() + underflow) = std::nextafter(
                edges.at(ax.size() + underflow), std::numeric_limits<double>::min());
        }

        return edges;
    };

    return select(
        continuous,
        [flow](const auto& ax) {
            using AX = std::decay_t<decltype(ax)>;
            static_assert(!bh::axis::traits::get_options<AX>::test(option::underflow),
                          "discrete axis never has underflow");

            const bh::axis::index_type overflow
                = flow && bh::axis::traits::get_options<AX>::test(option::overflow);

            py::array_t<double> edges(
                static_cast<py::ssize_t>(ax.size() + 1 + overflow));

            for(bh::axis::index_type i = 0; i <= ax.size() + overflow; ++i)
                edges.mutable_at(i) = i;

            return edges;
        },
        continuous,
        ax);
}

template <class A>
py::array_t<double> centers(const A& ax) {
    py::array_t<double> result(ax.size());
    for(bh::axis::index_type i = 0; i < ax.size(); ++i)
        result.mutable_data()[i] = unchecked_center(ax, i);
    return result;
}

template <class A>
py::array_t<double> widths(const A& ax) {
    py::array_t<double> result(ax.size());
    bh::detail::static_if<bh::axis::traits::is_continuous<A>>(
        [](py::array_t<double>& result, const auto& ax) {
            std::transform(ax.begin(),
                           ax.end(),
                           result.mutable_data(),
                           [](const auto& b) { return b.width(); });
        },
        [](py::array_t<double>& result, const auto& ax) {
            std::fill(result.mutable_data(), result.mutable_data() + ax.size(), 1.0);
        },
        result,
        ax);
    return result;
}

} // namespace axis

// The following list is all types supported
using axis_variant = bh::axis::variant<axis::regular_uoflow,
                                       axis::regular_uflow,
                                       axis::regular_oflow,
                                       axis::regular_none,
                                       axis::regular_uoflow_growth,
                                       axis::regular_circular,
                                       axis::regular_pow,
                                       axis::regular_trans,
                                       axis::regular_numpy,
                                       axis::variable_uoflow,
                                       axis::variable_uflow,
                                       axis::variable_oflow,
                                       axis::variable_none,
                                       axis::variable_uoflow_growth,
                                       axis::variable_circular,
                                       axis::integer_uoflow,
                                       axis::integer_uflow,
                                       axis::integer_oflow,
                                       axis::integer_none,
                                       axis::integer_growth,
                                       axis::integer_circular,
                                       axis::category_int,
                                       axis::category_int_growth,
                                       axis::category_str,
                                       axis::category_str_growth,
                                       axis::boolean>;

// This saves a little typing
using vector_axis_variant = std::vector<axis_variant>;

namespace pybind11 {
namespace detail {

/// Register axis_variant as a variant for pybind11
template <>
struct type_caster<axis_variant> : variant_caster<axis_variant> {};

} // namespace detail
} // namespace pybind11
