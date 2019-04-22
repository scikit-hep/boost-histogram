// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#include <boost/histogram/python/pybind11.hpp>

void register_version(py::module &);
py::module register_storages(py::module &);
py::module register_axes(py::module &);
py::module register_histograms(py::module &);
void register_make_histogram(py::module &, py::module &);
py::module register_accumulators(py::module &);

PYBIND11_MODULE(histogram, m) {
    register_version(m);
    register_storages(m);
    register_axes(m);
    py::module hist = register_histograms(m);
    register_make_histogram(m, hist);
    register_accumulators(m);
}
