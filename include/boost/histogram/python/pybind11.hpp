// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

// This file must be the first include

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <functional>
#include <type_traits>
#include <sstream>

#include <boost/mp11/list.hpp>
#include <boost/mp11/algorithm.hpp>

// Allow command line overide
#ifndef BOOST_HISTOGRAM_DETAIL_AXES_LIMIT
#define BOOST_HISTOGRAM_DETAIL_AXES_LIMIT 16
#endif

namespace py = pybind11;
using namespace pybind11::literals; // For ""_a syntax

namespace boost { namespace histogram {}}
namespace bh = boost::histogram;

/// Static if standin: define a method if expression is true
template<typename T, typename... Args>
void def_optionally(T&& module, std::true_type, Args&&... expression) {
    module.def(std::forward<Args...>(expression...));
}

/// Static if standin: Do nothing if compile time expression is false
template<typename T, typename... Args>
void def_optionally(T&&, std::false_type, Args&&...) {}

/// Shift to string
template<typename T>
auto shift_to_string() {
    return [](const T& self){
        std::ostringstream out;
        out << self;
        return out.str();
    };
}

// end of recursion
template <class R, class Unary>
R try_cast_impl(boost::mp11::mp_list<>, py::object, Unary&&) {
  throw py::cast_error("try_cast failed to find a match for object argument");
}

// recursion
template <class R, class T, class... Ts, class Unary>
R try_cast_impl(boost::mp11::mp_list<T, Ts...>, py::object obj, Unary&& unary) {
  if (py::isinstance<T>(obj))
    return unary(py::cast<T>(obj));
  return try_cast_impl<R>(boost::mp11::mp_list<Ts...>{}, obj, std::forward<Unary>(unary));
}

/**
  Cast python object to first match in type list and run functor with that type.

  Returns whatever the functor returns. The functor return type may not depend on the
  argument type.

  Throws pybind11::cast_error if no match is found.
*/
template <class TypeList, class Unary>
decltype(auto) try_cast(py::object obj, Unary&& unary) {
  using R = decltype(unary(std::declval<boost::mp11::mp_first<TypeList>>()));
  using L = boost::mp11::mp_rename<TypeList, boost::mp11::mp_list>;
  return try_cast_impl<R>(L{}, obj, std::forward<Unary>(unary));
}
