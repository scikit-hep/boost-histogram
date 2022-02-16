// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <bh_python/pybind11.hpp>

#include <bh_python/array_like.hpp>
#include <bh_python/axis.hpp>
#include <bh_python/fill.hpp>
#include <bh_python/make_pickle.hpp>

#include <boost/histogram/axis/ostream.hpp>
#include <boost/histogram/axis/traits.hpp>
#include <boost/histogram/detail/iterator_adaptor.hpp>
#include <boost/histogram/detail/span.hpp>

#include <pybind11/eval.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef __cpp_noexcept_function_type
#define BHP_NOEXCEPT_17 noexcept
#else
#define BHP_NOEXCEPT_17
#endif

template <class T>
auto vectorize_index(T input) {
    return py::vectorize(input);
}

#ifdef __cpp_noexcept_function_type
#define BHP_NOEXCEPT_17 noexcept
#else
#define BHP_NOEXCEPT_17
#endif

namespace detail {
template <class T>
decltype(auto) axis_cast(py::handle x) {
    return special_cast<T>(x);
}

template <>
inline decltype(auto) axis_cast<int>(py::handle x) {
    if(py::isinstance<int>(x))
        return py::cast<int>(x);

    auto val  = py::cast<float>(x);
    auto ival = static_cast<int>(val);

    if(static_cast<float>(ival) == val)
        return ival;

    throw py::type_error(py::str("cannot cast {} to int").format(val));
}
} // namespace detail

// we overload vectorize index for category axis
template <class T, class Options>
auto vectorize_index(int (bh::axis::category<T, metadata_t, Options>::*pindex)(const T&)
                         const BHP_NOEXCEPT_17) {
    return [pindex](const bh::axis::category<T, metadata_t, Options>& self,
                    py::object arg) -> py::object {
        auto index = std::mem_fn(pindex);

        if(detail::is_value<T>(arg)) {
            auto index_value = index(self, detail::axis_cast<T>(arg));
            if(index_value >= self.size())
                throw pybind11::key_error(py::str("{!r} not in axis").format(arg));
            return py::cast(index_value);
        }

        auto indices = array_like<int>(arg);
        auto values  = detail::special_cast<detail::c_array_t<T>>(arg);

        auto ip = indices.mutable_data();
        auto vp = values.data();
        for(std::size_t i = 0, n = values.size(); i < n; ++i) {
            ip[i] = index(self, vp[i]);
            if(ip[i] >= self.size())
                throw pybind11::key_error(py::str("{!r} not in axis").format(vp[i]));
        }

        return std::move(indices);
    };
}

template <class T>
auto vectorize_value(T input) {
    return py::vectorize(input);
}

// we overload vectorize value for category axis
template <class R, class U, class Options>
auto vectorize_value(R (bh::axis::category<U, metadata_t, Options>::*pvalue)(int)
                         const) {
    return [pvalue](const bh::axis::category<U, metadata_t, Options>& self,
                    py::object arg) -> py::object {
        auto value = std::mem_fn(pvalue);

        if(detail::is_value<int>(arg)) {
            auto i = py::cast<int>(arg);
            return i < self.size() ? py::cast(value(self, i)) : py::none();
        }

        auto indices = py::cast<py::array_t<int>>(arg);

        // this limitation could be removed if we find a way to make object arrays
        if(indices.ndim() != 1)
            throw std::invalid_argument("only ndim == 1 supported");

        const auto n = static_cast<std::size_t>(indices.size());
        py::tuple values(n);

        auto pi = indices.data();
        for(std::size_t k = 0; k < n; ++k) {
            const auto i = pi[k];
            unchecked_set(
                values, k, i < self.size() ? py::cast(value(self, i)) : py::none());
        }

        return std::move(values);
    };
}

#undef BHP_NOEXCEPT_17

/// Add helpers common to all axis types
template <class A, class... Args>
py::class_<A> register_axis(py::module& m, Args&&... args) {
    py::class_<A> ax(m, axis::string_name<A>(), std::forward<Args>(args)...);

    ax.def("__repr__", &shift_to_string<A>)

        .def("__eq__",
             [](const A& self, const py::object& other) {
                 try {
                     return self == py::cast<A>(other);
                 } catch(const py::cast_error&) {
                     return false;
                 }
             })
        .def("__ne__",
             [](const A& self, const py::object& other) {
                 try {
                     return self != py::cast<A>(other);
                 } catch(const py::cast_error&) {
                     return true;
                 }
             })

        .def_property_readonly("traits_underflow",
                               [](const A& self) {
                                   return static_cast<bool>(
                                       self.options() & bh::axis::option::underflow);
                               })
        .def_property_readonly("traits_overflow",
                               [](const A& self) {
                                   return static_cast<bool>(
                                       self.options() & bh::axis::option::overflow);
                               })
        .def_property_readonly("traits_circular",
                               [](const A& self) {
                                   return static_cast<bool>(
                                       self.options() & bh::axis::option::circular);
                               })
        .def_property_readonly("traits_growth",
                               [](const A& self) {
                                   return static_cast<bool>(self.options()
                                                            & bh::axis::option::growth);
                               })
        .def_property_readonly(
            "traits_continuous",
            [](const A& self) { return bh::axis::traits::continuous(self); })
        .def_property_readonly(
            "traits_ordered",
            [](const A& self) { return bh::axis::traits::ordered(self); })

        .def_property(
            "metadata",
            [](const A& self) { return self.metadata(); },
            [](A& self, const metadata_t& label) { self.metadata() = label; },
            "Set the axis label")

        .def_property_readonly(
            "size",
            [](const A& ob) { return ob.size(); },
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
                    = bh::axis::traits::get_options<A>::test(
                          bh::axis::option::underflow)
                          ? -1
                          : 0;
                const bh::axis::index_type end
                    = bh::axis::traits::get_options<A>::test(bh::axis::option::overflow)
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

        // This is a property because we hide the flow bins consistently here.
        // Users should use histogram.to_numpy(True) to get a consistent representation
        // of edges and the cell matrix, where we can guarantee that both are in sync.
        .def_property_readonly(
            "edges",
            [](const A& ax) { return axis::edges(ax, false); },
            "Return bin edges")
        .def_property_readonly("centers", &axis::centers<A>, "Return bin centers")
        .def_property_readonly("widths", &axis::widths<A>, "Return bin widths")

        .def("index",
             vectorize_index(&A::index),
             "Index for value (or values) on the axis")
        .def("value", vectorize_value(&A::value), "Value at index (or indices)")

        .def(make_pickle<A>());

    return ax;
}
