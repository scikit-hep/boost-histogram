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
                self(wei, val);
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
            self(py::cast<double>(weight), value);

        return self;
    };
}

void register_accumulators(py::module &accumulators) {
    using weighted_sum = bh::accumulators::weighted_sum<double>;

    register_accumulator<weighted_sum>(accumulators, "weighted_sum")

        .def(py::init<const double &>(), "value"_a)
        .def(py::init<const double &, const double &>(), "value"_a, "variance"_a)

        .def_property_readonly("variance", &weighted_sum::variance)
        .def_property_readonly("value", &weighted_sum::value)

        .def(py::self += double())

        .def("__repr__",
             [](const weighted_sum &self) {
                 return py::str("weighted_sum(value={}, variance={})")
                     .format(self.value(), self.variance());
             })

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

        ;

    using sum = bh::accumulators::sum<double>;

    register_accumulator<sum>(accumulators, "sum")
        .def(py::init<const double &>(), "value"_a)

        .def_property(
            "value", &sum::operator double, [](sum &s, double v) { s = v; })

        .def(py::self += double())

        .def("__repr__",
             [](const weighted_sum &self) {
                 return py::str("sum({})").format(double(self));
             })

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

    using weighted_mean = bh::accumulators::weighted_mean<double>;

    register_accumulator<weighted_mean>(accumulators, "weighted_mean")
        .def(py::init<const double &, const double &, const double &, const double &>(),
             "wsum"_a,
             "wsum2"_a,
             "value"_a,
             "variance"_a)

        .def_property_readonly("sum_of_weights", &weighted_mean::sum_of_weights)
        .def_property_readonly("variance", &weighted_mean::variance)
        .def_property_readonly("value", &weighted_mean::value)

        .def("__repr__",
             [](const weighted_mean &self) {
                 return py::str(
                            "weighted_mean(wsum={}, wsum2=..., mean={}, variance={})")
                     .format(self.sum_of_weights(), self.value(), self.variance());
             })

        .def("__call__",
             make_mean_call<weighted_mean>(),
             "value"_a,
             "Fill with value and optional keyword-only weight")

        .def("fill",
             make_mean_fill<weighted_mean>(),
             "value"_a,
             "Fill the accumulator with values. Optional weight parameter.")

        ;

    using mean = bh::accumulators::mean<double>;

    register_accumulator<mean>(accumulators, "mean")
        .def(py::init<const double &, const double &, const double &>(),
             "count"_a,
             "value"_a,
             "variance"_a)

        .def_property_readonly("count", &mean::count)
        .def_property_readonly("value", &mean::value)
        .def_property_readonly("variance", &mean::variance)

        .def("__call__",
             make_mean_call<mean>(),
             "value"_a,
             "Fill with value and optional keyword-only weight")

        .def("fill",
             make_mean_fill<mean>(),
             "value"_a,
             "Fill the accumulator with values. Optional weight parameter.")

        .def("__repr__",
             [](const mean &self) {
                 return py::str("mean(count={}, mean={}, variance={})")
                     .format(self.count(), self.value(), self.variance());
             })

        ;
}
