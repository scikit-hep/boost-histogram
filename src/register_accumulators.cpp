// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#include <boost/histogram/python/pybind11.hpp>

#include <boost/histogram/accumulators/sum.hpp>
#include <boost/histogram/python/accumulators/mean.hpp>
#include <boost/histogram/python/accumulators/weighted_mean.hpp>
#include <boost/histogram/python/accumulators/weighted_sum.hpp>
#include <boost/histogram/python/accumulators_ostream.hpp>
#include <boost/histogram/python/kwargs.hpp>
#include <boost/histogram/python/register_accumulator.hpp>
#include <pybind11/operators.h>

/// The mean fill can be implemented once. (sum fill varies slightly)
template <class T>
decltype(auto) make_mean_fill() {
    return [](T &self, py::object value, py::kwargs kwargs) {
        py::object weight = optional_arg(kwargs, "weight", py::none());
        finalize_args(kwargs);
        if(weight.is_none()) {
            py::vectorize([](T &self, double val) {
                self(val);
                return false;
            })(self, value);
        } else {
            py::vectorize([](T &self, double wei, double val) {
                self(bh::weight(wei), val);
                return false;
            })(self, weight, value);
        }
        return self;
    };
}

/// The mean call can be implemented once. (sum uses +=)
template <class T>
decltype(auto) make_mean_call() {
    return [](T &self, double value, py::kwargs kwargs) {
        py::object weight = optional_arg(kwargs, "weight", py::none());
        finalize_args(kwargs);

        if(weight.is_none())
            self(value);
        else
            self(bh::weight(py::cast<double>(weight)), value);

        return self;
    };
}

void register_accumulators(py::module &accumulators) {
    // Naming convention:
    // If a value is publically available in Boost.Histogram accumulators
    // as a method, it has the same name in the numpy record array.
    // If it is not available except through a computation, it has
    // the same name as the private property without the trailing _.

    using weighted_sum = accumulators::weighted_sum<double>;

    PYBIND11_NUMPY_DTYPE(weighted_sum, value, variance);

    register_accumulator<weighted_sum>(accumulators, "weighted_sum")

        .def(py::init<const double &>(), "value"_a)
        .def(py::init<const double &, const double &>(), "value"_a, "variance"_a)

        .def_readonly("value", &weighted_sum::value)
        .def_readonly("variance", &weighted_sum::variance)

        .def(py::self += double())

        .def(
            "fill",
            [](weighted_sum &self, py::object value, py::kwargs kwargs) {
                py::object variance = optional_arg(kwargs, "variance", py::none());
                finalize_args(kwargs);
                if(variance.is_none()) {
                    py::vectorize([](weighted_sum &self, double val) {
                        self += val;
                        return false;
                    })(self, value);
                } else {
                    py::vectorize([](weighted_sum &self, double val, double var) {
                        self += weighted_sum(val, var);
                        return false;
                    })(self, value, variance);
                }
                return self;
            },
            "value"_a,
            "Fill the accumulator with values. Optional variance parameter.")

        .def_static("_make", py::vectorize([](const double &a, const double &b) {
                        return weighted_sum(a, b);
                    }))

        ;

    using sum = bh::accumulators::sum<double>;

    register_accumulator<sum>(accumulators, "sum")
        .def(py::init<const double &>(), "value"_a)

        .def_property(
            "value", &sum::operator double, [](sum &s, double v) { s = v; })

        .def(py::self += double())

        .def(
            "fill",
            [](sum &self, py::object value) {
                py::vectorize([](sum &self, double v) {
                    self += v;
                    return false; // Required in PyBind11 2.4.2,
                                  // requirement may be removed
                })(self, value);
                return self;
            },
            "value"_a,
            "Run over an array with the accumulator")

        .def_property_readonly("small", &sum::small)
        .def_property_readonly("large", &sum::large)

        ;

    using weighted_mean = accumulators::weighted_mean<double>;
    PYBIND11_NUMPY_DTYPE(weighted_mean,
                         sum_of_weights,
                         sum_of_weights_squared,
                         value,
                         sum_of_weighted_deltas_squared);

    register_accumulator<weighted_mean>(accumulators, "weighted_mean")
        .def(py::init<const double &, const double &, const double &, const double &>(),
             "wsum"_a,
             "wsum2"_a,
             "value"_a,
             "variance"_a)

        .def_readonly("sum_of_weights", &weighted_mean::sum_of_weights)
        .def_readonly("sum_of_weights_squared", &weighted_mean::sum_of_weights_squared)
        .def_readonly("value", &weighted_mean::value)
        .def_readonly("sum_of_weighted_deltas_squared",
                      &weighted_mean::sum_of_weighted_deltas_squared)

        .def_property_readonly("variance", &weighted_mean::variance)

        .def("__call__",
             make_mean_call<weighted_mean>(),
             "value"_a,
             "Fill with value and optional keyword-only weight")

        .def("fill",
             make_mean_fill<weighted_mean>(),
             "value"_a,
             "Fill the accumulator with values. Optional weight parameter.")

        .def_static(
            "_make",
            py::vectorize(
                [](const double &a, const double &b, const double &c, double &d) {
                    return weighted_mean(a, b, c, d, true);
                }))

        ;

    using mean = accumulators::mean<double>;
    PYBIND11_NUMPY_DTYPE(mean, count, value, sum_of_deltas_squared);

    register_accumulator<mean>(accumulators, "mean")
        .def(py::init<const double &, const double &, const double &>(),
             "count"_a,
             "value"_a,
             "variance"_a)

        .def_readonly("count", &mean::count)
        .def_readonly("value", &mean::value)
        .def_readonly("sum_of_deltas_squared", &mean::sum_of_deltas_squared)

        .def_property_readonly("variance", &mean::variance)

        .def("__call__",
             make_mean_call<mean>(),
             "value"_a,
             "Fill with value and optional keyword-only weight")

        .def("fill",
             make_mean_fill<mean>(),
             "value"_a,
             "Fill the accumulator with values. Optional weight parameter.")

        .def_static(
            "_make",
            py::vectorize([](const double &a, const double &b, const double &c) {
                return mean(a, b, c, true);
            }))

        ;
}
