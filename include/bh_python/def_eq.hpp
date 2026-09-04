// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <bh_python/pybind11.hpp>

/// Compare with the GIL held (the default for def_eq)
struct keep_gil {};

/// Add __eq__ and __ne__ to a py::class_, comparing against another Python
/// object via A::operator==. Objects that cannot be cast to A compare
/// unequal rather than raising. Only operator== is required on A.
///
/// Pass py::gil_scoped_release as Gil to compare without the GIL, for a type
/// whose operator== is safe that way (a histogram: its axes reacquire the
/// GIL themselves for the metadata compare).
template <class Gil = keep_gil, class A, class... Extra>
py::class_<A, Extra...>& def_eq(py::class_<A, Extra...>& cls) {
    return cls
        .def("__eq__",
             [](const A& self, const py::object& other) {
                 try {
                     const A& other_ref = py::cast<const A&>(other);
                     const Gil gil;
                     (void)gil;
                     return self == other_ref;
                 } catch(const py::cast_error&) {
                     return false;
                 }
             })
        .def("__ne__", [](const A& self, const py::object& other) {
            try {
                const A& other_ref = py::cast<const A&>(other);
                const Gil gil;
                (void)gil;
                return !(self == other_ref);
            } catch(const py::cast_error&) {
                return true;
            }
        });
}
