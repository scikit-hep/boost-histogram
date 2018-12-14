// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "getpython.hpp"

void register_axis(py::module &);
void register_histogram(py::module &);

PYBIND11_MODULE(hist, m) {
    register_axis(m);
    register_histogram(m);
}
