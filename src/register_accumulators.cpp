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
        .def(init_1_vectorized_add<weighted_sum>(), "value"_a)

        .def(init_2_vectorized_add<weighted_sum>(), "value"_a, "variance"_a)

        .def_property_readonly("variance", &weighted_sum::variance)
        .def_property_readonly("value", &weighted_sum::value)

        .def(py::self += double())

        ;

    using sum = bh::accumulators::sum<double>;

    register_accumulator<sum>(accumulators, "sum")
        .def(init_1_vectorized_add<sum>(), "value"_a)

        .def_property(
            "value", &sum::operator double, [](sum &s, double v) { s = v; })

        .def(py::self += double())

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

        .def(init_1_vectorized_call<weighted_mean>(), "value"_a)

        .def(init_2_vectorized_call<weighted_mean>(), "weight"_a, "value"_a)

        .def_property_readonly("sum_of_weights", &weighted_mean::sum_of_weights)
        .def_property_readonly("variance", &weighted_mean::variance)
        .def_property_readonly("value", &weighted_mean::value)

        .def(
            "__call__",
            [](weighted_mean &self, double value) {
                self(value);
                return self;
            },
            "value"_a)
        .def(
            "__call__",
            [](weighted_mean &self, double weight, double value) {
                self(weight, value);
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

        .def(init_1_vectorized_call<mean>(), "value"_a)

        .def_property_readonly("count", &mean::count)
        .def_property_readonly("variance", &mean::variance)
        .def_property_readonly("value", &mean::value)

        .def(
            "__call__",
            [](mean &self, double value) {
                self(value);
                return self;
            },
            "value"_a)

        ;
}
