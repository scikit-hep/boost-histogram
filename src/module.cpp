// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#include <bh_python/pybind11.hpp>

void register_algorithms(py::module&);
void register_storages(py::module&);
void register_axes(py::module&);
void register_histograms(py::module&);
void register_accumulators(py::module&);
void register_transforms(py::module&);

PYBIND11_MODULE(_core, m) {
    py::module storage = m.def_submodule("storage");
    register_storages(storage);

    py::module ax = m.def_submodule("axis");
    register_axes(ax);

    py::module trans = ax.def_submodule("transform");
    register_transforms(trans);

    py::module hist = m.def_submodule("hist");
    register_histograms(hist);

    py::module accumulators = m.def_submodule("accumulators");
    register_accumulators(accumulators);

    py::module algorithm = m.def_submodule("algorithm");
    register_algorithms(algorithm);
}
