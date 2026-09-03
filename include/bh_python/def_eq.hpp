// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <bh_python/pybind11.hpp>

/// Add __eq__ and __ne__ to a py::class_, comparing against another Python
/// object via A::operator==. Objects that cannot be cast to A compare
/// unequal rather than raising. Only operator== is required on A.
template <class A, class... Extra>
py::class_<A, Extra...>& def_eq(py::class_<A, Extra...>& cls) {
    return cls
        .def("__eq__",
             [](const A& self, const py::object& other) {
                 try {
                     return self == py::cast<const A&>(other);
                 } catch(const py::cast_error&) {
                     return false;
                 }
             })
        .def("__ne__", [](const A& self, const py::object& other) {
            try {
                return !(self == py::cast<const A&>(other));
            } catch(const py::cast_error&) {
                return true;
            }
        });
}
