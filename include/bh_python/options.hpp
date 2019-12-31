// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <bh_python/pybind11.hpp>

#include <boost/histogram/axis/option.hpp>

struct options {
    unsigned option;

    options(unsigned value)
        : option(value) {}
    options(bool uflow, bool oflow, bool circ, bool grow)
        : option(
            uflow * bh::axis::option::underflow | oflow * bh::axis::option::overflow
            | circ * bh::axis::option::circular | grow * bh::axis::option::growth) {}

    bool operator==(const options& other) const { return option == other.option; }
    bool operator!=(const options& other) const { return option != other.option; }

    bool underflow() const {
        return static_cast<bool>(option & bh::axis::option::underflow);
    }
    bool overflow() const {
        return static_cast<bool>(option & bh::axis::option::overflow);
    }
    bool circular() const {
        return static_cast<bool>(option & bh::axis::option::circular);
    }
    bool growth() const { return static_cast<bool>(option & bh::axis::option::growth); }
};
