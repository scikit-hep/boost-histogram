// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <boost/histogram/python/pybind11.hpp>

#include <boost/variant2/variant.hpp>

/// Register boost::variant2::variant as a variant for PyBind11
namespace pybind11 {
namespace detail {
template <class... Ts>
struct type_caster<boost::variant2::variant<Ts...>> : variant_caster<boost::variant2::variant<Ts...>> {};
} // namespace detail
} // namespace pybind11
