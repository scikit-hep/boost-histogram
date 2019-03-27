// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#include <boost/histogram/python/pybind11.hpp>

void register_axis(py::module &);
void register_histogram(py::module &);
void register_storage(py::module &);
void register_accumulators(py::module &);

PYBIND11_MODULE(histogram, m) {
    register_storage(m);
    register_axis(m);
    register_histogram(m);
    register_accumulators(m);
}
