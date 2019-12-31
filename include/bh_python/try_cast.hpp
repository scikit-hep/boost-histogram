// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <bh_python/pybind11.hpp>

#include <boost/mp11/algorithm.hpp>
#include <boost/mp11/list.hpp>

#include <utility>

// end of recursion
template <class R, class Unary>
R try_cast_impl(boost::mp11::mp_list<>, py::object, Unary&&) {
    throw py::cast_error("try_cast failed to find a match for object argument");
}

// recursion
template <class R, class T, class... Ts, class Unary>
R try_cast_impl(boost::mp11::mp_list<T, Ts...>, py::object obj, Unary&& unary) {
    if(py::isinstance<T>(obj))
        return unary(py::cast<T>(obj));
    return try_cast_impl<R>(
        boost::mp11::mp_list<Ts...>{}, obj, std::forward<Unary>(unary));
}

/**
  Cast python object to first match in type list and run functor with that type.

  Returns whatever the functor returns. The functor return type must not depend on
  the argument type. In other words, all functor overloads must return the same type.

  Throws pybind11::cast_error if no match is found.
*/
template <class TypeList, class Unary>
decltype(auto) try_cast_over(py::object obj, Unary&& unary) {
    using R = decltype(unary(std::declval<boost::mp11::mp_first<TypeList>>()));
    using L = boost::mp11::mp_rename<TypeList, boost::mp11::mp_list>;
    return try_cast_impl<R>(L{}, obj, std::forward<Unary>(unary));
}

/// Like try_cast_over, but passing the types explicitly.
template <class T, class... Ts, class Unary>
decltype(auto) try_cast(py::object obj, Unary&& unary) {
    return try_cast_over<boost::mp11::mp_list<T, Ts...>>(obj,
                                                         std::forward<Unary>(unary));
}
