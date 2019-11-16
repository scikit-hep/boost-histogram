// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#include <boost/histogram/python/pybind11.hpp>

#include <pybind11/operators.h>

#include <boost/histogram/axis/regular.hpp>

template <class T, class... Args>
py::class_<T> register_transform(py::module &mod, Args &&... args) {
    py::class_<T> transform(mod, std::forward<Args>(args)...);
    transform.def(py::init<T>());
    transform.def("forward", [](const T &self, double v) { return self.forward(v); });
    transform.def("inverse", [](const T &self, double v) { return self.inverse(v); });
    return transform;
}

void register_transforms(py::module &mod) {
    register_transform<bh::axis::transform::id>(mod, "id")
        .def(py::init<>())
        .def("__repr__",
             [](py::object self) {
                 return py::str("{}()").format(self.attr("__class__").attr("__name__"));
             })

        ;

    register_transform<bh::axis::transform::sqrt>(mod, "sqrt")
        .def(py::init<>())
        .def("__repr__",
             [](py::object self) {
                 return py::str("{}()").format(self.attr("__class__").attr("__name__"));
             })

        ;

    register_transform<bh::axis::transform::log>(mod, "log")
        .def(py::init<>())
        .def("__repr__",
             [](py::object self) {
                 return py::str("{}()").format(self.attr("__class__").attr("__name__"));
             })

        ;

    register_transform<bh::axis::transform::pow>(mod, "pow")
        .def(py::init<double>())
        .def_readonly("power", &bh::axis::transform::pow::power)
        .def("__repr__",
             [](py::object self) {
                 double power = py::cast<bh::axis::transform::pow>(self).power;
                 return py::str("{}({:g})")
                     .format(self.attr("__class__").attr("__name__"), power);
             })

        ;
}
