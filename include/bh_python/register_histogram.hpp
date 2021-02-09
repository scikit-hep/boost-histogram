// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <bh_python/pybind11.hpp>

#include <bh_python/accumulators/ostream.hpp>
#include <bh_python/axis.hpp>
#include <bh_python/fill.hpp>
#include <bh_python/histogram.hpp>
#include <bh_python/make_pickle.hpp>
#include <bh_python/storage.hpp>

#include <boost/histogram/algorithm/empty.hpp>
#include <boost/histogram/algorithm/project.hpp>
#include <boost/histogram/algorithm/reduce.hpp>
#include <boost/histogram/algorithm/sum.hpp>
#include <boost/histogram/histogram.hpp>
#include <boost/histogram/ostream.hpp>
#include <boost/histogram/unsafe_access.hpp>
#include <boost/mp11.hpp>

#include <future>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

class Ticker {
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> indices_;
    size_t n_;
    size_t linear_size_;

  public:
    Ticker(std::vector<size_t> shape)
        : shape_(shape)
        , indices_(shape.size(), 0)
        , n_(0)
        , linear_size_(std::accumulate(
              std::begin(shape), std::end(shape), 1u, std::multiplies<std::size_t>())) {
    }

    std::size_t ndim() const { return shape_.size(); }
    std::size_t linear_size() const { return linear_size_; }
    const std::vector<std::size_t>& shape() const { return shape_; }
    const std::vector<std::size_t>& indices() const { return indices_; }

    void next() {
        for(std::size_t i = ndim() - 1; i >= 0; --i) {
            n_++;
            indices_[i]++;
            if(indices_[i] != shape_[i])
                break;
            indices_[i] = 0;
        }
    }
    bool done() const { return n_ < linear_size_; }
};

/// Like py::vectorize, but for dynamic inputs (py::args)
/// There are inputs.size() number of args / function arguments, using the letter i for
/// looping. There are result.ndim() number of diminsions in the broadcast arrays, using
/// j for looping
template <class T, class V>
py::array_t<T> dynamic_vectorize(std::function<T(const std::vector<V>&&)> func,
                                 const py::args& args) {
    std::vector<py::array_t<V>> inputs;
    for(const auto&& arg : args) {
        try {
            inputs.push_back(py::cast<py::array_t<V>>(arg));
        } catch(const py::cast_error&) {
            inputs.emplace_back({
                py::cast<V>(arg),
            });
        }
    }

    // Prepare the shape vector; holds the broadcast shape
    std::vector<std::size_t> shape;
    for(const py::array_t<V>&& input : inputs) {
        for(std::size_t i = 0; i < input.ndim(); i++) {
            if(shape.size() < i)
                shape.emplace_back(1);

            if(shape[i] == 1)
                shape[i] = input.shape(i);
            else if(shape[i] != input.shape(i))
                throw std::runtime_error("Missmatched shapes!");
        }
    }

    std::vector<V> input_vector(inputs.size());
    py::array_t<T> result{shape, {}};

    for(Ticker ticker{shape}; !ticker.done(); ticker.next()) {
        for(std::size_t i; i < inputs.size(); ++i) {
            for(std::size_t j; j < ticker.ndim(); j++) {
                size_t idx = ticker.indices()[j];
                if(j >= inputs[i].ndim() || inputs[i].shape(j) == 1)
                    idx = 1;
                input_vector[i] = inputs[i].at(idx);
            }
        }
        result(i) = func(input_vector); // i needs to be correct
    }

    return result;
}

template <class S>
auto register_histogram(py::module& m, const char* name, const char* desc) {
    using histogram_t = bh::histogram<vector_axis_variant, S>;
    using value_type  = typename histogram_t::value_type;

    py::class_<histogram_t> hist(m, name, desc, py::buffer_protocol());

    hist.def(py::init<const vector_axis_variant&, S>(), "axes"_a, "storage"_a = S())

        .def_buffer(
            [](histogram_t& h) -> py::buffer_info { return make_buffer(h, false); })

        .def("rank", &histogram_t::rank)
        .def("size", &histogram_t::size)
        .def("reset", &histogram_t::reset)

        .def("__copy__", [](const histogram_t& self) { return histogram_t(self); })
        .def("__deepcopy__",
             [](const histogram_t& self, py::object memo) {
                 auto* a         = new histogram_t(self);
                 py::module copy = py::module::import("copy");
                 for(unsigned i = 0; i < a->rank(); i++) {
                     bh::unsafe_access::axis(*a, i).metadata()
                         = copy.attr("deepcopy")(a->axis(i).metadata(), memo);
                 }
                 return a;
             })

        .def(py::self += py::self)

        .def("__eq__",
             [](const histogram_t& self, const py::object& other) {
                 try {
                     return self == py::cast<histogram_t>(other);
                 } catch(const py::cast_error&) {
                     return false;
                 }
             })
        .def("__ne__",
             [](const histogram_t& self, const py::object& other) {
                 try {
                     return self != py::cast<histogram_t>(other);
                 } catch(const py::cast_error&) {
                     return true;
                 }
             })

        .def_property_readonly_static(
            "_storage_type",
            [](py::object) {
                return py::type::of<typename histogram_t::storage_type>();
            })

        ;

    def_optionally(hist,
                   bh::detail::has_operator_rdiv<histogram_t, histogram_t>{},
                   py::self /= py::self);
    def_optionally(hist,
                   bh::detail::has_operator_rmul<histogram_t, histogram_t>{},
                   py::self *= py::self);

    hist.def(
            "to_numpy",
            [](histogram_t& h, bool flow) {
                py::tuple tup(1 + h.rank());

                // Add the histogram buffer as the first argument
                unchecked_set(tup, 0, py::array(make_buffer(h, flow)));

                // Add the axis edges
                h.for_each_axis([&tup, flow, i = 0u](const auto& ax) mutable {
                    unchecked_set(tup, ++i, axis::edges(ax, flow, true));
                });

                return tup;
            },
            "flow"_a = false)

        .def(
            "view",
            [](py::object self, bool flow) {
                auto& h = py::cast<histogram_t&>(self);
                return py::array(make_buffer(h, flow), self);
            },
            "flow"_a = false)

        .def(
            "axis",
            [](const histogram_t& self, int i) -> py::object {
                unsigned ii = i < 0 ? self.rank() - static_cast<unsigned>(std::abs(i))
                                    : static_cast<unsigned>(i);

                if(ii < self.rank()) {
                    const axis_variant& var = self.axis(ii);
                    return bh::axis::visit(
                        [](auto&& item) -> py::object {
                            // Here we return a new, no-copy py::object that
                            // is not yet tied to the histogram. py::keep_alive
                            // is needed to make sure the histogram is alive as long
                            // as the axes references are.
                            return py::cast(item, py::return_value_policy::reference);
                        },
                        var);

                }

                else
                    throw std::out_of_range(
                        "The axis value must be less than the rank");
            },
            "i"_a = 0,
            py::keep_alive<0, 1>())

        .def("at",
             [](const histogram_t& self, py::args& args) -> value_type {
                 auto int_args = py::cast<std::vector<int>>(args);
                 return self.at(int_args);
             })

        .def("_at_set",
             [](histogram_t& self, const value_type& input, py::args& args) {
                 auto int_args     = py::cast<std::vector<int>>(args);
                 self.at(int_args) = input;
             })

        .def("__repr__", &shift_to_string<histogram_t>)

        .def(
            "sum",
            [](const histogram_t& self, bool flow) {
                return bh::algorithm::sum(
                    self, flow ? bh::coverage::all : bh::coverage::inner);
            },
            "flow"_a = false)

        .def(
            "empty",
            [](const histogram_t& self, bool flow) {
                return bh::algorithm::empty(
                    self, flow ? bh::coverage::all : bh::coverage::inner);
            },
            "flow"_a = false)

        .def("reduce",
             [](const histogram_t& self, py::args args) {
                 return bh::algorithm::reduce(
                     self, py::cast<std::vector<bh::algorithm::reduce_command>>(args));
             })

        .def("project",
             [](const histogram_t& self, py::args values) {
                 return bh::algorithm::project(self,
                                               py::cast<std::vector<unsigned>>(values));
             })

        .def("fill", &fill<histogram_t>)

        .def(make_pickle<histogram_t>())

        ;

    return hist;
}
