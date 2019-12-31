// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#include <bh_python/pybind11.hpp>

#include <pybind11/functional.h>
#include <pybind11/operators.h>

#include <bh_python/axis.hpp>
#include <bh_python/make_pickle.hpp>
#include <bh_python/transform.hpp>
#include <boost/histogram/axis/regular.hpp>

template <class T, class... Args>
py::class_<T> register_transform(py::module& mod, Args&&... args) {
    py::class_<T> transform(mod, std::forward<Args>(args)...);

    transform

        .def(py::init<T>())
        .def("forward", [](const T& self, double v) { return self.forward(v); })
        .def("inverse", [](const T& self, double v) { return self.inverse(v); })
        .def(make_pickle<T>())
        .def("__copy__", [](const T& self) { return T(self); })
        .def("__deepcopy__", &deep_copy<T>)

        ;

    return transform;
}

extern "C" {
double _log_fn(double v) { return std::log(v); }
double _exp_fn(double v) { return std::exp(v); }
double _sqrt_fn(double v) { return std::sqrt(v); }
double _sq_fn(double v) { return v * v; }
}

void register_transforms(py::module& mod) {
    mod.def("_log_fn", &_log_fn);
    mod.def("_exp_fn", &_exp_fn);
    mod.def("_sqrt_fn", &_sqrt_fn);
    mod.def("_sq_fn", &_sq_fn);

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
        .def(py::init<double>(), "power"_a)
        .def_readonly("power", &bh::axis::transform::pow::power)
        .def("__repr__",
             [](py::object self) {
                 double power = py::cast<bh::axis::transform::pow>(self).power;
                 return py::str("{}({:g})")
                     .format(self.attr("__class__").attr("__name__"), power);
             })

        ;

    register_transform<func_transform>(mod, "func_transform")
        .def(py::init<py::object, py::object, py::object, py::str>(),
             "forward"_a,
             "inverse"_a,
             "convert"_a,
             "name"_a)
        .def("__repr__",
             [](py::object self) {
                 auto& s = py::cast<func_transform&>(self);
                 if(s._name.equal(py::str(""))) {
                     return py::str("{}({}, {})")
                         .format(self.attr("__class__").attr("__name__"),
                                 s._forward_ob,
                                 s._inverse_ob);
                 } else
                     return s._name;
             })

        ;
}
