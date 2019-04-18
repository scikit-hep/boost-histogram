// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <boost/histogram.hpp>

#include <boost/histogram/python/pybind11.hpp>

#include <tuple>
#include <vector>

/// Register bh::axis::variant as a variant for PyBind11
namespace pybind11 {
namespace detail {
template <typename... Ts>
struct type_caster<bh::axis::variant<Ts...>> : variant_caster<bh::axis::variant<Ts...>> {};
} // namespace detail
} // namespace pybind11

/// Utility to convert an axis to edges array
template <typename A>
py::array_t<double> axis_to_edges(const A &ax, bool flow) {
    unsigned overflow  = flow && (bh::axis::traits::options(ax) & bh::axis::option::underflow);
    unsigned underflow = flow && (bh::axis::traits::options(ax) & bh::axis::option::overflow);

    py::array_t<double> edges((unsigned)ax.size() + 1u + overflow + underflow);

    if(underflow)
        edges.mutable_at(0u) = ax.bin(-1).lower();

    edges.mutable_at(0u + underflow) = ax.bin(0).lower();

    std::transform(
        ax.begin(), ax.end(), edges.mutable_data() + 1u + underflow, [](const auto &bin) { return bin.upper(); });

    if(overflow)
        edges.mutable_at(edges.size() - 1) = ax.bin(ax.size()).upper();

    return edges;
}

template <typename A>
decltype(auto) axis_to_bins(const A &self, bool flow) {
    std::vector<bh::detail::remove_cvref_t<decltype(self.bin(0))>> out;
    bool overflow  = flow && (bh::axis::traits::options(self) & bh::axis::option::underflow);
    bool underflow = flow && (bh::axis::traits::options(self) & bh::axis::option::overflow);

    out.reserve((size_t)bh::axis::traits::extent(self));

    for(int i = 0 - underflow; i < self.size() + overflow; i++)
        out.emplace_back(self.bin(i));

    return out;
}

inline bool PyObject_Check(void *value) { return value != nullptr; }

#if PY_MAJOR_VERSION < 3

extern PyThreadState *_PyThreadState_Current;

inline int PyGILState_Check(void) {
    PyThreadState *tstate = _PyThreadState_Current;
    return tstate && (tstate == PyGILState_GetThisThreadState());
}
#endif

class metadata_t : public py::object {
    PYBIND11_OBJECT_DEFAULT(metadata_t, object, PyObject_Check);

    bool operator==(const metadata_t &other) const {
        if(PyGILState_Check()) {
            return py::cast<bool>(this->attr("__eq__")(other));
        } else {
            py::gil_scoped_acquire gil;
            return py::cast<bool>(this->attr("__eq__")(other));
        }
    }
    bool operator!=(const metadata_t &other) const {
        if(PyGILState_Check()) {
            return py::cast<bool>(this->attr("__ne__")(other));
        } else {
            py::gil_scoped_acquire gil;
            return py::cast<bool>(this->attr("__ne__")(other));
        }
    }
};

namespace axis {

// These match the Python names
using regular_uoflow      = bh::axis::regular<double, bh::use_default, metadata_t>;
using regular_noflow      = bh::axis::regular<double, bh::use_default, metadata_t, bh::axis::option::none_t>;
using regular_growth      = bh::axis::regular<double, bh::use_default, metadata_t, bh::axis::option::growth_t>;
using circular            = bh::axis::circular<double, metadata_t>;
using regular_log         = bh::axis::regular<double, bh::axis::transform::log, metadata_t>;
using regular_sqrt        = bh::axis::regular<double, bh::axis::transform::sqrt, metadata_t>;
using regular_pow         = bh::axis::regular<double, bh::axis::transform::pow, metadata_t>;
using variable            = bh::axis::variable<double, metadata_t>;
using integer_uoflow      = bh::axis::integer<int, metadata_t>;
using integer_noflow      = bh::axis::integer<int, metadata_t, bh::axis::option::none_t>;
using integer_growth      = bh::axis::integer<int, metadata_t, bh::axis::option::growth_t>;
using category_int        = bh::axis::category<int, metadata_t>;
using category_int_growth = bh::axis::category<int, metadata_t, bh::axis::option::growth_t>;
using category_str        = bh::axis::category<std::string, metadata_t>;
using category_str_growth = bh::axis::category<std::string, metadata_t, bh::axis::option::growth_t>;

} // namespace axis

namespace axes {

// The following list is all types supported
using any = std::vector<bh::axis::variant<axis::regular_uoflow,
                                          axis::regular_noflow,
                                          axis::regular_growth,
                                          axis::circular,
                                          axis::regular_log,
                                          axis::regular_pow,
                                          axis::regular_sqrt,
                                          axis::variable,
                                          axis::integer_uoflow,
                                          axis::integer_noflow,
                                          axis::integer_growth,
                                          axis::category_int,
                                          axis::category_int_growth,
                                          axis::category_str,
                                          axis::category_str_growth>>;

// Specialization for some speed improvement
using regular = std::vector<axis::regular_uoflow>;

// Specialization for some speed improvement
using regular_noflow = std::vector<axis::regular_noflow>;

} // namespace axes
