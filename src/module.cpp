// Copyright 2015-2018 Hans Dembinski and Henry Schreiner
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/histogram/python/pybind11.hpp>


void register_axis(py::module &);
void register_histogram(py::module &);
void register_storage(py::module &);

PYBIND11_MODULE(boosthistogram, m) {
    register_storage(m);
    register_axis(m);
    register_histogram(m);
}
