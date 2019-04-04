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
