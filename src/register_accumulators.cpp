// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#include <bh_python/pybind11.hpp>

#include <bh_python/accumulators/mean.hpp>
#include <bh_python/accumulators/ostream.hpp>
#include <bh_python/accumulators/weighted_mean.hpp>
#include <bh_python/accumulators/weighted_sum.hpp>
#include <bh_python/kwargs.hpp>
#include <bh_python/register_accumulator.hpp>
#include <boost/histogram/accumulators/sum.hpp>
#include <pybind11/operators.h>

/// The mean fill can be implemented once. (sum fill varies slightly)
template <class T>
decltype(auto) make_mean_fill() {
    return [](T& self, py::object value, py::object weight) {
        if(weight.is_none()) {
            py::vectorize([](T& self, double val) { self(val); })(self, value);
        } else {
            py::vectorize([](T& self, double wei, double val) {
                self(bh::weight(wei), val);
            })(self, weight, value);
        }
        return self;
    };
}

/// The mean call can be implemented once. (sum uses +=)
template <class T>
decltype(auto) make_mean_call() {
    return [](T& self, double value, py::object weight) {
        if(weight.is_none())
            self(value);
        else
            self(bh::weight(py::cast<double>(weight)), value);

        return self;
    };
}

template <class T>
decltype(auto) make_buffer() {
    return [](T& self) -> py::buffer_info {
        return py::buffer_info(
            &self,                              // Pointer to buffer
            sizeof(T),                          // Size of one scalar
            py::format_descriptor<T>::format(), // Format registered with
                                                // PYBIND11_NUMPY_DTYPE
            0,                                  // Number of dimensions
            {},                                 // Buffer dimensions
            {}                                  // Stride
        );
    };
}

void register_accumulators(py::module& accumulators) {
    // Naming convention:
    // If a value is publically available in Boost.Histogram accumulators
    // as a method, it has the same name in the numpy record array.
    // If it is not available except through a computation, it has
    // the same name as the private property without the trailing _.

    using weighted_sum = accumulators::weighted_sum<double>;

    PYBIND11_NUMPY_DTYPE(weighted_sum, value, variance);

    register_accumulator<weighted_sum>(
        accumulators, "WeightedSum", py::buffer_protocol())

        .def_buffer(make_buffer<weighted_sum>())

        .def(py::init<const double&>(), "value"_a)
        .def(py::init<const double&, const double&>(), "value"_a, "variance"_a)

        .def_readonly("value", &weighted_sum::value)
        .def_readonly("variance", &weighted_sum::variance)

        .def("__iadd__",
             [](weighted_sum& self, double value) {
                 self += bh::weight(value);
                 return self;
             })

        .def(
            "fill",
            [](weighted_sum& self, py::object value, py::object variance) {
                if(variance.is_none()) {
                    py::vectorize([](weighted_sum& self, double val) {
                        self += bh::weight(val);
                    })(self, value);
                } else {
                    py::vectorize([](weighted_sum& self, double val, double var) {
                        self += weighted_sum(val, var);
                    })(self, value, variance);
                }
                return self;
            },
            "value"_a,
            py::kw_only(),
            "variance"_a = py::none(),
            "Fill the accumulator with values. Optional variance parameter.")

        .def_static("_make", py::vectorize([](const double& a, const double& b) {
                        return weighted_sum(a, b);
                    }))

        .def("__getitem__",
             [](const weighted_sum& self, py::str key) {
                 if(key.equal(py::str("value")))
                     return self.value;
                 else if(key.equal(py::str("variance")))
                     return self.variance;
                 else
                     throw py::key_error(
                         py::str("{0} not one of value, variance").format(key));
             })
        .def("__setitem__",
             [](weighted_sum& self, py::str key, double value) {
                 if(key.equal(py::str("value")))
                     self.value = value;
                 else if(key.equal(py::str("variance")))
                     self.variance = value;
                 else
                     throw py::key_error(
                         py::str("{0} not one of value, variance").format(key));
             })

        .def("_ipython_key_completions_",
             [](py::object /* self */) { return py::make_tuple("value", "variance"); })

        ;

    using sum = bh::accumulators::sum<double>;

    register_accumulator<sum>(accumulators, "Sum")
        .def(py::init<const double&>(), "value"_a)

        .def_property_readonly("value", &sum::value)

        .def(py::self += double())

        .def(
            "fill",
            [](sum& self, py::object value) {
                py::vectorize([](sum& self, double v) { self += v; })(self, value);
                return self;
            },
            "value"_a,
            "Run over an array with the accumulator")

        .def_property_readonly("_small", &sum::small)
        .def_property_readonly("_large", &sum::large)

        ;

    using weighted_mean = accumulators::weighted_mean<double>;
    PYBIND11_NUMPY_DTYPE(weighted_mean,
                         sum_of_weights,
                         sum_of_weights_squared,
                         value,
                         _sum_of_weighted_deltas_squared);

    register_accumulator<weighted_mean>(
        accumulators, "WeightedMean", py::buffer_protocol())

        .def_buffer(make_buffer<weighted_mean>())

        .def(py::init<const double&, const double&, const double&, const double&>(),
             "sum_of_weights"_a,
             "sum_of_weights_squared"_a,
             "value"_a,
             "variance"_a)

        .def_readonly("sum_of_weights", &weighted_mean::sum_of_weights)
        .def_readonly("sum_of_weights_squared", &weighted_mean::sum_of_weights_squared)
        .def_readonly("value", &weighted_mean::value)
        .def_readonly("_sum_of_weighted_deltas_squared",
                      &weighted_mean::_sum_of_weighted_deltas_squared)

        .def_property_readonly("variance", &weighted_mean::variance)

        .def("__call__",
             make_mean_call<weighted_mean>(),
             "value"_a,
             py::kw_only(),
             "weight"_a = py::none(),
             "Fill with value and optional keyword-only weight")

        .def("fill",
             make_mean_fill<weighted_mean>(),
             "value"_a,
             py::kw_only(),
             "weight"_a = py::none(),
             "Fill the accumulator with values. Optional weight parameter.")

        .def_static(
            "_make",
            py::vectorize(
                [](const double& a, const double& b, const double& c, double& d) {
                    return weighted_mean(a, b, c, d, true);
                }))

        .def("__getitem__",
             [](const weighted_mean& self, py::str key) {
                 if(key.equal(py::str("value")))
                     return self.value;
                 else if(key.equal(py::str("sum_of_weights")))
                     return self.sum_of_weights;
                 else if(key.equal(py::str("sum_of_weights_squared")))
                     return self.sum_of_weights_squared;
                 else if(key.equal(py::str("_sum_of_weighted_deltas_squared")))
                     return self._sum_of_weighted_deltas_squared;
                 else
                     throw py::key_error(
                         py::str(
                             "{0} not one of value, sum_of_weights, "
                             "sum_of_weights_squared, _sum_of_weighted_deltas_squared")
                             .format(key));
             })
        .def("__setitem__",
             [](weighted_mean& self, py::str key, double value) {
                 if(key.equal(py::str("value")))
                     self.value = value;
                 else if(key.equal(py::str("sum_of_weights")))
                     self.sum_of_weights = value;
                 else if(key.equal(py::str("sum_of_weights_squared")))
                     self.sum_of_weights_squared = value;
                 else if(key.equal(py::str("_sum_of_weighted_deltas_squared")))
                     self._sum_of_weighted_deltas_squared = value;
                 else
                     throw py::key_error(
                         py::str(
                             "{0} not one of value, sum_of_weights, "
                             "sum_of_weights_squared, _sum_of_weighted_deltas_squared")
                             .format(key));
             })

        .def("_ipython_key_completions_",
             [](py::object /* self */) {
                 return py::make_tuple("value",
                                       "sum_of_weights",
                                       "sum_of_weights_squared",
                                       "_sum_of_weighted_deltas_squared");
             })

        ;

    using mean = accumulators::mean<double>;
    PYBIND11_NUMPY_DTYPE(mean, count, value, sum_of_deltas_squared);

    register_accumulator<mean>(accumulators, "Mean", py::buffer_protocol())
        .def_buffer(make_buffer<mean>())

        .def(py::init<const double&, const double&, const double&>(),
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
             py::kw_only(),
             "weight"_a = py::none(),
             "Fill with value and optional keyword-only weight")

        .def("fill",
             make_mean_fill<mean>(),
             "value"_a,
             py::kw_only(),
             "weight"_a = py::none(),
             "Fill the accumulator with values. Optional weight parameter.")

        .def_static(
            "_make",
            py::vectorize([](const double& a, const double& b, const double& c) {
                return mean(a, b, c, true);
            }))

        .def("__getitem__",
             [](const mean& self, py::str key) {
                 if(key.equal(py::str("count")))
                     return self.count;
                 else if(key.equal(py::str("value")))
                     return self.value;
                 else if(key.equal(py::str("sum_of_deltas_squared")))
                     return self.sum_of_deltas_squared;
                 else
                     throw py::key_error(
                         py::str("{0} not one of count, value, sum_of_deltas_squared")
                             .format(key));
             })
        .def("__setitem__",
             [](mean& self, py::str key, double value) {
                 if(key.equal(py::str("count")))
                     self.count = value;
                 else if(key.equal(py::str("value")))
                     self.value = value;
                 else if(key.equal(py::str("sum_of_deltas_squared")))
                     self.sum_of_deltas_squared = value;
                 else
                     throw py::key_error(
                         py::str("{0} not one of count, value, sum_of_deltas_squared")
                             .format(key));
             })

        .def("_ipython_key_completions_",
             [](py::object /* self */) {
                 return py::make_tuple("count", "value", "sum_of_deltas_squared");
             })

        ;
}
