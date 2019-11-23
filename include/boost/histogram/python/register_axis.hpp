// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <boost/histogram/python/pybind11.hpp>

#include <boost/histogram/axis/ostream.hpp>
#include <boost/histogram/axis/traits.hpp>
#include <boost/histogram/detail/iterator_adaptor.hpp>
#include <boost/histogram/detail/span.hpp>
#include <boost/histogram/python/axis.hpp>
#include <boost/histogram/python/make_pickle.hpp>
#include <boost/histogram/python/options.hpp>

#include <algorithm>
#include <functional>
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

// we overload vectorize index for category axis
template <class Options>
auto vectorize(int (bh::axis::category<std::string, metadata_t, Options>::*pindex)(
    const std::string&) const) {
    return [pindex](const bh::axis::category<std::string, metadata_t, Options>& self,
                    py::object arg) -> py::object {
        auto index = std::mem_fn(pindex);
        if(py::isinstance<py::str>(arg))
            return py::cast(index(self, py::cast<std::string>(arg)));

        if(py::isinstance<py::array>(arg)) {
            auto arr = py::cast<py::array>(arg);
            if(arr.ndim() != 1)
                throw std::invalid_argument("only ndim == 1 supported");
        }

        auto values = py::cast<py::sequence>(arg);
        py::array_t<int> indices(values.size());

        auto ip = indices.mutable_data();
        for(auto&& v : values) {
            if(!py::isinstance<py::str>(v))
                throw std::invalid_argument("input is not a string");
            *ip++ = index(self, py::cast<std::string>(v));
        }

        return std::move(indices);
    };
}

// we overload vectorize value for category axis
template <class Result, class U, class Options>
auto vectorize(Result (bh::axis::category<U, metadata_t, Options>::*pvalue)(int)
                   const) {
    return
        [pvalue](
            const std::conditional_t<std::is_same<U, std::string>::value,
                                     axis::category_str_t<Options>,
                                     bh::axis::category<U, metadata_t, Options>>& self,
            py::object arg) -> py::object {
            auto value = std::mem_fn(pvalue);

            if(py::isinstance<py::int_>(arg)) {
                auto i = py::cast<int>(arg);
                return i < self.size() ? py::cast(value(self, i)) : py::none();
            }

            if(py::isinstance<py::array>(arg)) {
                auto arr = py::cast<py::array>(arg);
                if(arr.ndim() != 1)
                    throw std::invalid_argument("only ndim == 1 supported");
            }

            auto indices = py::cast<py::sequence>(arg);
            py::tuple values(indices.size());

            unsigned k = 0;
            for(auto&& ipy : indices) {
                const auto i = py::cast<int>(ipy);
                unchecked_set(values,
                              k++,
                              i < self.size() ? py::cast(value(self, i)) : py::none());
            }

            return std::move(values);
        };
}

template <class A>
void vectorized_index_and_value_methods(py::class_<A>& axis) {
    // find our vectorize and pybind11::vectorize
    using ::vectorize;
    using py::vectorize;
    axis.def("index",
             vectorize(&A::index),
             "Index for value (or values) on the axis",
             "x"_a)
        .def("value", vectorize(&A::value), "Value at index (or indices)", "i"_a);
}

/// Add helpers common to all axis types
template <class A, class... Args>
py::class_<A> register_axis(py::module& m, Args&&... args) {
    py::class_<A> ax(m, axis::string_name<A>(), std::forward<Args>(args)...);

    ax.def("__repr__", &shift_to_string<A>)

        .def(py::self == py::self)
        .def(py::self != py::self)

        .def_property_readonly(
            "options",
            [](const A& self) {
                return options{static_cast<unsigned>(self.options())};
            },
            "Return the options associated to the axis")

        .def_property(
            "metadata",
            [](const A& self) { return self.metadata(); },
            [](A& self, const metadata_t& label) { self.metadata() = label; },
            "Set the axis label")

        .def_property_readonly(
            "size",
            &A::size,
            "Returns the number of bins excluding under- and overflow")

        .def_property_readonly(
            "extent",
            &bh::axis::traits::extent<A>,
            "Returns the number of bins including under- and overflow")

        .def("__copy__", [](const A& self) { return A(self); })
        .def("__deepcopy__",
             [](const A& self, py::object memo) {
                 A* a            = new A(self);
                 py::module copy = py::module::import("copy");
                 a->metadata()   = copy.attr("deepcopy")(a->metadata(), memo);
                 return a;
             })

        .def(
            "bin",
            [](const A& ax, int i) {
                const bh::axis::index_type begin
                    = bh::axis::traits::static_options<A>::test(
                          bh::axis::option::underflow)
                          ? -1
                          : 0;
                const bh::axis::index_type end
                    = bh::axis::traits::static_options<A>::test(
                          bh::axis::option::overflow)
                          ? ax.size() + 1
                          : ax.size();
                if(begin <= i && i < end)
                    return axis::unchecked_bin<A>(ax, i);
                throw py::index_error();
            },
            "i"_a,
            "Return bin at index (-1 accesses underflow bin, size access overflow)")

        .def(
            "__iter__",
            [](const A& ax) {
                struct iterator
                    : bh::detail::iterator_adaptor<iterator, int, py::object> {
                    const A& axis_;
                    iterator(const A& axis, int idx)
                        : iterator::iterator_adaptor_(idx)
                        , axis_(axis) {}

                    auto operator*() const {
                        return axis::unchecked_bin<A>(axis_, this->base());
                    }
                };

                iterator begin(ax, 0), end(ax, ax.size());
                return py::make_iterator(begin, end);
            },
            py::keep_alive<0, 1>())

        .def_property_readonly(
            "edges",
            [](const A& ax) { return axis::edges(ax, false); },
            "Return bin edges")
        .def_property_readonly("centers", &axis::centers<A>, "Return bin centers")
        .def_property_readonly("widths", &axis::widths<A>, "Return bin widths");

    vectorized_index_and_value_methods(ax);

    ax.def(make_pickle<A>());

    return ax;
}
