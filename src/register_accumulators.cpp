// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#include <bh_python/pybind11.hpp>

#include <bh_python/accumulators/mean.hpp>
#include <bh_python/accumulators/ostream.hpp>
#include <bh_python/accumulators/weighted_collector.hpp>
#include <bh_python/accumulators/weighted_mean.hpp>
#include <bh_python/accumulators/weighted_sum.hpp>
#include <bh_python/kwargs.hpp>
#include <bh_python/make_pickle.hpp>
#include <bh_python/register_accumulator.hpp>
#include <boost/histogram/accumulators/collector.hpp>
#include <boost/histogram/accumulators/sum.hpp>
#include <pybind11/operators.h>

#include <utility>
#include <vector>

namespace {
/// The mean fill can be implemented once. (sum fill varies slightly)
template <class T>
decltype(auto) make_mean_fill() {
    return [](T& self, const py::object& value, const py::object& weight) {
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
    return [](T& self, double value, const py::object& weight) {
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

} // namespace

// NOLINTNEXTLINE(misc-use-internal-linkage)
void register_accumulators(py::module& accumulators) {
    // Naming convention:
    // If a value is publicly available in Boost.Histogram accumulators
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
            [](weighted_sum& self,
               const py::object& value,
               const py::object& variance) {
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

        // This adapts existing memory to an accumulator
        .def_static("_make", py::vectorize([](const double& a, const double& b) {
                        return weighted_sum(a, b);
                    }))

        // This creates a array of accumulators using the normal constructor arguments
        .def_static("_array", py::vectorize([](const double& a, const double& b) {
                        return weighted_sum(a, b);
                    }))

        .def("__getitem__",
             [](const weighted_sum& self, const py::str& key) {
                 if(key.equal(py::str("value")))
                     return self.value;
                 if(key.equal(py::str("variance")))
                     return self.variance;
                 throw py::key_error(
                     py::str("{0} not one of value, variance").format(key));
             })
        .def("__setitem__",
             [](weighted_sum& self, const py::str& key, double value) {
                 if(key.equal(py::str("value")))
                     self.value = value;
                 else if(key.equal(py::str("variance")))
                     self.variance = value;
                 else
                     throw py::key_error(
                         py::str("{0} not one of value, variance").format(key));
             })

        .def("_ipython_key_completions_",
             [](const py::object& /* self */) {
                 return py::make_tuple("value", "variance");
             })

        ;

    using sum = bh::accumulators::sum<double>;

    register_accumulator<sum>(accumulators, "Sum")
        .def(py::init<const double&>(), "value"_a)

        .def_property_readonly("value", &sum::value)

        .def(py::self += double())

        .def(
            "fill",
            [](sum& self, const py::object& value) {
                py::vectorize([](sum& self, double v) { self += v; })(self, value);
                return self;
            },
            "value"_a,
            "Run over an array with the accumulator")

        .def_property_readonly("_small", &sum::small_part)
        .def_property_readonly("_large", &sum::large_part)

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
                [](const double& a, const double& b, const double& c, const double& d) {
                    return weighted_mean(a, b, c, d, true);
                }))

        .def_static(
            "_array",
            py::vectorize(
                [](const double& a, const double& b, const double& c, const double& d) {
                    return weighted_mean(a, b, c, d);
                }))

        .def("__getitem__",
             [](const weighted_mean& self, const py::str& key) {
                 if(key.equal(py::str("value")))
                     return self.value;
                 if(key.equal(py::str("sum_of_weights")))
                     return self.sum_of_weights;
                 if(key.equal(py::str("sum_of_weights_squared")))
                     return self.sum_of_weights_squared;
                 if(key.equal(py::str("_sum_of_weighted_deltas_squared")))
                     return self._sum_of_weighted_deltas_squared;
                 throw py::key_error(
                     py::str("{0} not one of value, sum_of_weights, "
                             "sum_of_weights_squared, _sum_of_weighted_deltas_squared")
                         .format(key));
             })
        .def("__setitem__",
             [](weighted_mean& self, const py::str& key, double value) {
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
             [](const py::object& /* self */) {
                 return py::make_tuple("value",
                                       "sum_of_weights",
                                       "sum_of_weights_squared",
                                       "_sum_of_weighted_deltas_squared");
             })

        ;

    using mean = accumulators::mean<double>;
    PYBIND11_NUMPY_DTYPE(mean, count, value, _sum_of_deltas_squared);

    register_accumulator<mean>(accumulators, "Mean", py::buffer_protocol())
        .def_buffer(make_buffer<mean>())

        .def(py::init<const double&, const double&, const double&>(),
             "count"_a,
             "value"_a,
             "variance"_a)

        .def_readonly("count", &mean::count)
        .def_readonly("value", &mean::value)
        .def_readonly("_sum_of_deltas_squared", &mean::_sum_of_deltas_squared)

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

        .def_static(
            "_array",
            py::vectorize([](const double& a, const double& b, const double& c) {
                return mean(a, b, c);
            }))

        .def("__getitem__",
             [](const mean& self, const py::str& key) {
                 if(key.equal(py::str("count")))
                     return self.count;
                 if(key.equal(py::str("value")))
                     return self.value;
                 if(key.equal(py::str("_sum_of_deltas_squared")))
                     return self._sum_of_deltas_squared;
                 throw py::key_error(
                     py::str("{0} not one of count, value, _sum_of_deltas_squared")
                         .format(key));
             })
        .def("__setitem__",
             [](mean& self, const py::str& key, double value) {
                 if(key.equal(py::str("count")))
                     self.count = value;
                 else if(key.equal(py::str("value")))
                     self.value = value;
                 else if(key.equal(py::str("_sum_of_deltas_squared")))
                     self._sum_of_deltas_squared = value;
                 else
                     throw py::key_error(
                         py::str("{0} not one of count, value, _sum_of_deltas_squared")
                             .format(key));
             })

        .def("_ipython_key_completions_",
             [](const py::object& /* self */) {
                 return py::make_tuple("count", "value", "_sum_of_deltas_squared");
             })

        ;

    // The per-bin cells of the (Weighted)Collector storages, exposed as their own
    // sequence accumulators: a vector of collected samples. h[i] and sum() return
    // these. Elements are plain floats / (value, weight) tuples; .value (and .weight)
    // give the columns as numpy arrays. __len__ + __getitem__ make them iterable.
    using values = bh::accumulators::collector<std::vector<double>>;

    py::class_<values>(accumulators, "Values")
        .def(py::init<>())
        .def(py::init([](const std::vector<double>& v) { return values(v); }),
             "values"_a)

        .def("__len__", [](const values& self) { return self.size(); })
        .def("__getitem__",
             [](const values& self, py::ssize_t i) {
                 const auto n = static_cast<py::ssize_t>(self.size());
                 if(i < 0)
                     i += n;
                 if(i < 0 || i >= n)
                     throw py::index_error();
                 return self[static_cast<std::size_t>(i)];
             })

        .def_property_readonly("value",
                               [](const values& self) {
                                   return py::array_t<double>(
                                       static_cast<py::ssize_t>(self.size()),
                                       self.data());
                               })

        .def("__eq__",
             [](const values& self, const py::object& other) {
                 try {
                     return self == py::cast<const values&>(other);
                 } catch(const py::cast_error&) {
                     return false;
                 }
             })
        .def("__ne__",
             [](const values& self, const py::object& other) {
                 try {
                     return self != py::cast<const values&>(other);
                 } catch(const py::cast_error&) {
                     return true;
                 }
             })

        .def("__repr__",
             [](const py::object& self) {
                 const auto& item = py::cast<const values&>(self);
                 py::list items;
                 for(const auto& v : item)
                     items.append(v);
                 return py::str("{0.__class__.__name__}({1})").format(self, items);
             })

        .def("__copy__", [](const values& self) { return values(self); })
        .def("__deepcopy__",
             [](const values& self, const py::object&) { return values(self); })

        .def(make_pickle<values>());

    using weighted_values = accumulators::weighted_collector<double>;

    py::class_<weighted_values>(accumulators, "WeightedValues")
        .def(py::init<>())
        .def(py::init([](const std::vector<std::pair<double, double>>& v) {
                 weighted_values::container_type cont;
                 cont.reserve(v.size());
                 for(const auto& p : v)
                     cont.emplace_back(p.first, p.second);
                 return weighted_values(std::move(cont));
             }),
             "values"_a)

        .def("__len__", [](const weighted_values& self) { return self.size(); })
        .def("__getitem__",
             [](const weighted_values& self, py::ssize_t i) {
                 const auto n = static_cast<py::ssize_t>(self.size());
                 if(i < 0)
                     i += n;
                 if(i < 0 || i >= n)
                     throw py::index_error();
                 const auto& e = self[static_cast<std::size_t>(i)];
                 return py::make_tuple(e.value, e.weight);
             })

        .def_property_readonly("value",
                               [](const weighted_values& self) {
                                   py::array_t<double> arr(
                                       static_cast<py::ssize_t>(self.size()));
                                   double* out = arr.mutable_data();
                                   for(const auto& e : self)
                                       *out++ = e.value;
                                   return arr;
                               })
        .def_property_readonly("weight",
                               [](const weighted_values& self) {
                                   py::array_t<double> arr(
                                       static_cast<py::ssize_t>(self.size()));
                                   double* out = arr.mutable_data();
                                   for(const auto& e : self)
                                       *out++ = e.weight;
                                   return arr;
                               })

        .def("__eq__",
             [](const weighted_values& self, const py::object& other) {
                 try {
                     return self == py::cast<const weighted_values&>(other);
                 } catch(const py::cast_error&) {
                     return false;
                 }
             })
        .def("__ne__",
             [](const weighted_values& self, const py::object& other) {
                 try {
                     return self != py::cast<const weighted_values&>(other);
                 } catch(const py::cast_error&) {
                     return true;
                 }
             })

        .def("__repr__",
             [](const py::object& self) {
                 const auto& item = py::cast<const weighted_values&>(self);
                 py::list items;
                 for(const auto& e : item)
                     items.append(py::make_tuple(e.value, e.weight));
                 return py::str("{0.__class__.__name__}({1})").format(self, items);
             })

        .def("__copy__",
             [](const weighted_values& self) { return weighted_values(self); })
        .def("__deepcopy__",
             [](const weighted_values& self, const py::object&) {
                 return weighted_values(self);
             })

        .def(make_pickle<weighted_values>());
}
