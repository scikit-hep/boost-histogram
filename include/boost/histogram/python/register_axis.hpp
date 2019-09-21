// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <boost/histogram/python/pybind11.hpp>

#include <boost/histogram/axis/traits.hpp>
#include <boost/histogram/detail/cat.hpp>
#include <boost/histogram/python/axis.hpp>
#include <boost/histogram/python/axis_ostream.hpp>
#include <boost/histogram/python/bin_setup.hpp>
#include <boost/histogram/python/serializion.hpp>

#include <algorithm>
#include <iostream>
#include <pybind11/eval.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace std::literals;

/// Ensure metadata is valid.
inline void validate_metadata(metadata_t v) {
    try {
        py::cast<double>(v);
        throw py::type_error(
            "Numeric types not allowed for metatdata in the the constructor. Use "
            ".metadata after "
            "constructing instead if you *really* need numeric metadata.");
    } catch(const py::cast_error &) {
    }
}

/// Add a constructor for an axes, with smart handling for the metadata (will not allow
/// numeric types)"
template <class T, class... Args>
decltype(auto) construct_axes() {
    return py::init([](Args... args, metadata_t v) {
        // Check for numeric v here
        validate_metadata(v);
        return new T{std::forward<Args>(args)..., v};
    });
}

template <class Iterable>
inline auto max_string_length(const Iterable &c) {
    std::string::size_type n = 0;
    for(auto &&s : c)
        n = std::max(n, s.size());
    return n;
}

template <class A>
void vectorized_index_and_value_methods(py::class_<A> &axis) {
    axis.def("index",
             py::vectorize(&A::index),
             "Index for value (or values) on the axis",
             "x"_a)
        .def("value", py::vectorize(&A::value), "Value at index (or indices)", "i"_a);
}

template <class... Ts>
void vectorized_index_and_value_methods(
    py::class_<bh::axis::category<int, Ts...>> &axis) {
    using axis_t = bh::axis::category<int, Ts...>;
    axis.def("index",
             py::vectorize([](axis_t &self, int v) { return int(self.index(v)); }),
             "Index for value (or values) on the axis",
             "x"_a)
        .def("value",
             py::vectorize([](axis_t &self, int i) { return int(self.value(i)); }),
             "Value at index (or indices)",
             "i"_a);
}

template <class... Ts>
void vectorized_index_and_value_methods(
    py::class_<bh::axis::category<std::string, Ts...>> &axis) {
    using axis_t = bh::axis::category<std::string, Ts...>;
    axis.def(
            "index",
            [](axis_t &self, py::array values) {
                py::array_t<int> indices(
                    bh::detail::span<const ssize_t>(values.shape(),
                                                    values.shape() + values.ndim()),
                    bh::detail::span<const ssize_t>(values.strides(),
                                                    values.strides() + values.ndim()));
                if(values.dtype().kind() != 'S')
                    throw std::invalid_argument(
                        "argument must be convertible to array of strings");
                const auto itemsize    = values.itemsize();
                auto pvalues           = static_cast<const char *>(values.data());
                const auto pvalues_end = pvalues + itemsize * values.size();
                auto *pindices         = indices.mutable_data();
                for(; pvalues != pvalues_end; pvalues += itemsize)
                    *pindices++ = self.index(std::string(pvalues, pvalues + itemsize));
                return indices;
            },
            "Index for value (or values) on the axis",
            "x"_a)
        .def(
            "value",
            [](axis_t &self, py::array_t<int> indices) {
                // we return an array of python objects, because the items we return
                // are highly redundant
                py::array values(
                    py::dtype("O"),
                    bh::detail::span<const ssize_t>(indices.shape(),
                                                    indices.shape() + indices.ndim()),
                    bh::detail::span<const ssize_t>(
                        indices.strides(), indices.strides() + indices.ndim()));
                // prepare vector of pre-converted python strings
                std::vector<py::str> strings(self.begin(), self.end());
                auto pindices           = indices.data();
                const auto pindices_end = pindices + indices.size();
                auto pvalues = static_cast<PyObject **>(values.mutable_data());
                for(; pindices != pindices_end; ++pindices)
                    *pvalues++ = strings.at(static_cast<unsigned>(*pindices)).ptr();
                return values;
            },
            "Value at index (or indices)",
            "i"_a);
}

/// Add helpers common to all axis types
template <class A, class... Args>
py::class_<A> register_axis(py::module &m, const char *name, Args &&... args) {
    py::class_<A> axis(m, name, std::forward<Args>(args)...);

    axis.def("__repr__", shift_to_string<A>())

        .def(py::self == py::self)
        .def(py::self != py::self)

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

        .def("extent",
             &bh::axis::traits::extent<A>,
             "Returns the number of bins with over- or underflow")

        .def("__len__", &axis::size())
        .def("__getitem__", axis::bin)

        ;

    bh::detail::static_if<axis::is_continuous<A>>(
        [](auto &axis) {
            // for continuous axis with bins that represent intervals
            axis.def("edges",
                     axis::to_edges,
                     "flow"_a = false,
                     "Bin edges (length: len(axis) + 1) (include over/underflow if "
                     "flow=True)")
                .def("centers", axis::to_centers, "Return the bin centers");
        },
        [](auto &axis) {
            // for discrete axis with bins that represent values
            axis.def(
                "values", axis::to_values, "flow"_a = false, "Return the bin values");
        },
        axis);

    vectorized_index_and_value_methods(axis);

    axis.def(make_pickle<A>());

    return axis;
}
