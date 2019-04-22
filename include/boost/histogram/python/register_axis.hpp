// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <boost/histogram/python/pybind11.hpp>

#include <pybind11/eval.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>

#include <boost/histogram/python/axis.hpp>
#include <boost/histogram/python/pickle.hpp>

#include <boost/histogram.hpp>
#include <boost/histogram/axis/ostream.hpp>
#include <boost/histogram/axis/traits.hpp>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

using namespace std::literals;

/// Ensure metadata is valid.
inline void validate_metadata(metadata_t v) {
    try {
        py::cast<double>(v);
        throw py::type_error("Numeric types not allowed for metatdata in the the constructor. Use .metadata after "
                             "constructing instead if you *really* need numeric metadata.");
    } catch(const py::cast_error &) {
    }
}

/// Add a constructor for an axes, with smart handling for the metadata (will not allow numeric types)"
template <typename T, typename... Args>
decltype(auto) construct_axes() {
    return py::init([](Args... args, metadata_t v) {
        // Check for numeric v here
        validate_metadata(v);
        return new T{std::forward<Args>(args)..., v};
    });
}

/// Add helpers common to all types with a range of values
template <typename A>
py::class_<bh::axis::interval_view<A>> register_axis_iv(py::module &m, const char *name) {
    using A_iv               = bh::axis::interval_view<A>;
    py::class_<A_iv> axis_iv = py::class_<A_iv>(m, name, "Lightweight bin view");

    axis_iv.def("upper", &A_iv::upper)
        .def("lower", &A_iv::lower)
        .def("center", &A_iv::center)
        .def("width", &A_iv::width)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("__repr__", [](const A_iv &self) {
            return "<bin ["s + std::to_string(self.lower()) + ", "s + std::to_string(self.upper()) + "]>"s;
        });

    return axis_iv;
}

/// Add items to an axis where the axis values are continious
template <typename A, typename B>
void add_to_axis(py::module m, const char *name, B &&axis, std::false_type) {
    axis.def("bin", &A::bin, "The bin details (center, lower, upper)", "idx"_a, py::keep_alive<0, 1>());
    axis.def(
        "bins", [](const A &self, bool flow) { return axis_to_bins(self, flow); }, "flow"_a = false);
    axis.def("index", py::vectorize(&A::index), "The index at a point(s) on the axis", "x"_a);
    axis.def("value", py::vectorize(&A::value), "The value(s) for a fractional bin(s) in the axis", "i"_a);

    axis.def(
        "edges",
        [](const A &ax, bool flow) { return axis_to_edges(ax, flow); },
        "flow"_a = false,
        "The bin edges (length: bins + 1) (include over/underflow if flow=True)");

    axis.def(
        "centers",
        [](const A &ax) {
            py::array_t<double> centers((unsigned)ax.size());
            std::transform(ax.begin(), ax.end(), centers.mutable_data(), [](const auto &bin) { return bin.center(); });
            return centers;
        },
        "Return the bin centers");

    std::string name_iv = "_"s + name + "_iv"s;

    register_axis_iv<A>(m, name_iv.c_str());
}

/// Add items to an axis where the axis values are not continious (categories of strings, for example)
template <typename A, typename B>
void add_to_axis(py::module, const char *, B &&axis, std::true_type) {
    axis.def("bin", &A::bin, "The bin name", "idx"_a);
    axis.def(
        "bins", [](const A &self, bool flow) { return axis_to_bins(self, flow); }, "flow"_a = false);
    // Not that these really just don't work with string labels; they would work for numerical labels.
    axis.def("index", &A::index, "The index at a point on the axis", "x"_a);
    axis.def("value", &A::value, "The value for a fractional bin in the axis", "i"_a);
}

/// Add helpers common to all axis types
template <typename A, typename... Args>
py::class_<A> register_axis(py::module &m, const char *name, Args &&... args) {
    py::class_<A> axis(m, name, std::forward<Args>(args)...);

    // using value_type = decltype(A::value(1.0));

    axis.def("__repr__", shift_to_string<A>())

        .def("__eq__", [](const A &self, const A &other) { return compare_axes_eq(self, other); })
        .def("__ne__", [](const A &self, const A &other) { return compare_axes_ne(self, other); })

        // Fall through for non-matching types
        .def("__eq__", [](const A &, const py::object &) { return false; })
        .def("__ne__", [](const A &, const py::object &) { return true; })

        .def(
            "size",
            [](const A &self, bool flow) {
                if(flow)
                    return bh::axis::traits::extent(self);
                else
                    return self.size();
            },
            "flow"_a = false,
            "Returns the number of bins, without over- or underflow unless flow=True")

        .def("update", &A::update, "Bin and add a value if allowed", "i"_a)
        .def_static("options", &A::options, "Return the options associated to the axis")
        .def_property(
            "metadata",
            [](const A &self) { return self.metadata(); },
            [](A &self, const metadata_t &label) { self.metadata() = label; },
            "Set the axis label")

        .def("__copy__", [](const A &self) { return A(self); })
        .def("__deepcopy__",
             [](const A &self, py::object memo) {
                 A *a            = new A(self);
                 py::module copy = py::module::import("copy");
                 a->metadata()   = copy.attr("deepcopy")(a->metadata(), memo);
                 return a;
             })

        ;

    // We only need keepalive if this is a reference.
    using Result = decltype(std::declval<A>().bin(std::declval<int>()));

    // This is a replacement for constexpr if
    add_to_axis<A>(m,
                   name,
                   axis,
                   std::integral_constant < bool,
                   std::is_reference<Result>::value || std::is_integral<Result>::value > {});

    axis.def(make_pickle<A>());

    return axis;
}
