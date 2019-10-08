// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <boost/histogram/python/pybind11.hpp>

#include <boost/histogram/accumulators/ostream.hpp> // TODO: replace with internal repr
#include <boost/histogram/python/serializion.hpp>

#include <pybind11/operators.h>
#include <utility>

template <class A, class... Args>
py::class_<A> register_accumulator(py::module acc, Args &&... args) {
    return py::class_<A>(acc, std::forward<Args>(args)...)
        .def(py::init<>())

        .def(py::self += py::self)
        .def(py::self == py::self)
        .def(py::self != py::self)

        .def(py::self *= double())

        .def("__repr__", &shift_to_string<A>)

        .def("__copy__", [](const A &self) { return A(self); })
        .def("__deepcopy__", [](const A &self, py::object) { return A(self); })

        .def(make_pickle<A>());
}

template <class T>
decltype(auto) init_1_vectorized_add() {
    return py::init([](py::object input_1) {
        auto ptr = new T();

        // There is an alternate way to do this using py::array_t<double>
        // and would skip allocating a useless bool array. However, it is
        // only 20% faster and does not handle broadcasting and scalars.
        // Let's piggyback on py::vectorize for now to get those nice features.
        //
        // // This is just as fast as using raw pointers, and more elegant
        // auto r = input_1.unchecked<1>();
        // for(py::ssize_t idx = 0; idx < r.shape(0); ++idx)
        //     (*ptr) += r(idx);

        py::vectorize([](T *p, const double &in_1) {
            (*p) += in_1;
            return false; // Needed for py::vectorize output
        })(ptr, input_1);
        return ptr;
    });
}

template <class T>
decltype(auto) init_1_vectorized_call() {
    return py::init([](py::object input_1) {
        auto ptr = new T();

        py::vectorize([](T *p, const double &in_1) {
            (*p)(in_1);
            return false; // Needed for py::vectorize output
        })(ptr, input_1);
        return ptr;
    });
}

template <class T>
decltype(auto) init_2_vectorized_add() {
    return py::init([](py::object input_1, py::object input_2) {
        auto ptr = new T();

        py::vectorize([](T *p, const double &in_1, const double &in_2) {
            (*p) += T(in_1, in_2);
            return false; // Needed for py::vectorize output
        })(ptr, input_1, input_2);
        return ptr;
    });
}

template <class T>
decltype(auto) init_2_vectorized_call() {
    return py::init([](py::object input_1, py::object input_2) {
        auto ptr = new T();

        py::vectorize([](T *p, const double &in_1, const double &in_2) {
            (*p)(in_1, in_2);
            return false; // Needed for py::vectorize output
        })(ptr, input_1, input_2);
        return ptr;
    });
}
