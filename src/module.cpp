// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#include <boost/histogram/python/pybind11.hpp>

void register_version(py::module &);
void register_storages(py::module &);
void register_axes(py::module &);
void register_axes_options(py::module &);
void register_general_histograms(py::module &);
void register_make_histogram(py::module &, py::module &);
void register_accumulators(py::module &);

PYBIND11_MODULE(histogram, m) {
    register_version(m);

    py::module storage = m.def_submodule("storage");
    register_storages(storage);

    py::module ax  = m.def_submodule("axis");
    py::module opt = ax.def_submodule("options");
    register_axes_options(opt);
    register_axes(ax);

    py::module hist = m.def_submodule("hist");
    register_general_histograms(hist);
    register_make_histogram(m, hist);

    py::module accumulators = m.def_submodule("accumulators");
    register_accumulators(accumulators);
}
