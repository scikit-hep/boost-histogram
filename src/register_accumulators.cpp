// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#include <boost/histogram/python/pybind11.hpp>

#include <boost/histogram/accumulators/mean.hpp>
#include <boost/histogram/accumulators/ostream.hpp>
#include <boost/histogram/accumulators/sum.hpp>
#include <boost/histogram/accumulators/weighted_mean.hpp>
#include <boost/histogram/accumulators/weighted_sum.hpp>
#include <boost/histogram/python/register_accumulator.hpp>
#include <pybind11/operators.h>

void register_accumulators(py::module &accumulators) {
    using weighted_sum = bh::accumulators::weighted_sum<double>;

    register_accumulator<weighted_sum>(accumulators, "weighted_sum")

        .def(py::init<const double &>(), "value"_a)
        .def(py::init<const double &, const double &>(), "value"_a, "variance"_a)

        .def_property_readonly("variance", &weighted_sum::variance)
        .def_property_readonly("value", &weighted_sum::value)

        .def(py::self += double())
        .def("__iadd__",
             [](weighted_sum &self, py::array_t<double> arr) {
                 auto r = arr.unchecked<1>();
                 for(py::ssize_t idx = 0; idx < r.shape(0); ++idx)
                     self += r(idx);
                 return self;
             })

        // Note: there is no way to do a vectorized sum with weights.
        // Constructors that take a vector of inputs would be one solution.

        ;

    using sum = bh::accumulators::sum<double>;

    register_accumulator<sum>(accumulators, "sum")
        .def(py::init<const double &>(), "value"_a)

        .def_property(
            "value", &sum::operator double, [](sum &s, double v) { s = v; })

        .def(py::self += double())
        .def("__iadd__",
             [](sum &self, py::array_t<double> arr) {
                 auto r = arr.unchecked<1>();
                 for(py::ssize_t idx = 0; idx < r.shape(0); ++idx)
                     self += r(idx);
                 return self;
             })

        .def_property_readonly("small", &sum::small)
        .def_property_readonly("large", &sum::large)

        ;

    using weighted_mean = bh::accumulators::weighted_mean<double>;

    register_accumulator<weighted_mean>(accumulators, "weighted_mean")
        .def(py::init<const double &, const double &, const double &, const double &>(),
             "wsum"_a,
             "wsum2"_a,
             "mean"_a,
             "variance"_a)

        .def_property_readonly("sum_of_weights", &weighted_mean::sum_of_weights)
        .def_property_readonly("variance", &weighted_mean::variance)
        .def_property_readonly("value", &weighted_mean::value)

        .def(
            "__call__",
            [](weighted_mean &self, py::object value) {
                py::vectorize([](weighted_mean &self, double v) {
                    self(v);
                    return false; // Required for PyBind11 for now (may be fixed
                                  // after 2.4.2)
                })(self, value);
                return self;
            },
            "value"_a)
        .def(
            "__call__",
            [](weighted_mean &self, py::object weight, py::object value) {
                py::vectorize([](weighted_mean &self, double w, double v) {
                    self(w, v);
                    return false;
                })(self, weight, value);
                return self;
            },
            "weight"_a,
            "value"_a)

        ;

    using mean = bh::accumulators::mean<double>;

    register_accumulator<mean>(accumulators, "mean")
        .def(py::init<std::size_t, const double &, const double &>(),
             "value"_a,
             "mean"_a,
             "variance"_a)

        .def_property_readonly("count", &mean::count)
        .def_property_readonly("variance", &mean::variance)
        .def_property_readonly("value", &mean::value)

        .def(
            "__call__",
            [](mean &self, py::object value) {
                py::vectorize([](mean &self, double v) {
                    self(v);
                    return false;
                })(self, value);
                return self;
            },
            "value"_a)
        .def(
            "__call__",
            [](mean &self, py::object weight, py::object value) {
                py::vectorize([](mean &self, double w, double x) {
                    self(w, x);
                    return false;
                })(self, weight, value);
                return self;
            },
            "weight"_a,
            "value"_a)

        ;
}
