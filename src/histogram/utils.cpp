// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#include <boost/histogram/python/pybind11.hpp>

struct loc {
    double value;
};

struct rebin {
    unsigned factor;
};

struct project {};

void register_utils(py::module &m) {
    py::class_<loc>(m, "loc").def(py::init<double>()).def_readwrite("value", &loc::value);

    py::class_<rebin>(m, "rebin")
        .def(py::init<unsigned>())
        .def_property_readonly_static("projection", [](py::object /* self */) { return false; })
        .def_readwrite("factor", &rebin::factor);

    py::class_<project>(m, "_project").def(py::init<>()).def_property_readonly_static("projection", []() {
        return true;
    });

    m.attr("project") = project{};
}
