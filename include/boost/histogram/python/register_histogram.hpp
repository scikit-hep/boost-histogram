// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <boost/histogram/python/kwargs.hpp>
#include <boost/histogram/python/pybind11.hpp>

#include <boost/histogram/algorithm/empty.hpp>
#include <boost/histogram/algorithm/project.hpp>
#include <boost/histogram/algorithm/reduce.hpp>
#include <boost/histogram/algorithm/sum.hpp>
#include <boost/histogram/histogram.hpp>
#include <boost/histogram/python/axis.hpp>
#include <boost/histogram/python/histogram.hpp>
#include <boost/histogram/python/histogram_ostream.hpp>
#include <boost/histogram/python/kwargs.hpp>
#include <boost/histogram/python/make_pickle.hpp>
#include <boost/histogram/python/storage.hpp>
#include <boost/histogram/python/sum.hpp>
#include <boost/histogram/python/variant.hpp>
#include <boost/histogram/unsafe_access.hpp>
#include <boost/mp11.hpp>
#include <future>
#include <pybind11/operators.h>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace detail {
template <class T, class... Us>
using is_one_of = boost::mp11::mp_contains<boost::mp11::mp_list<Us...>, T>;

template <class T>
bool is_pyiterable(const T &t) {
    return py::isinstance<py::buffer>(t) || py::hasattr(t, "__iter__");
}

template <class T, class VArg, class Arg>
void set_varg(boost::mp11::mp_identity<T>, VArg &v, const Arg &x) {
    if(is_pyiterable(x)) {
        auto arr
            = py::cast<py::array_t<T, py::array::c_style | py::array::forcecast>>(x);
        if(arr.ndim() != 1)
            throw std::invalid_argument("All arrays must be 1D");
        v = arr;
    } else
        v = py::cast<T>(x);
}

// specialization for string (HD: this is very inefficient and will be made more
// efficient in the future)
template <class VArg, class Arg>
void set_varg(boost::mp11::mp_identity<std::string>, VArg &v, const Arg &x) {
    if(py::isinstance<py::str>(x))
        v = py::cast<std::string>(x);
    else
        v = py::cast<std::vector<std::string>>(x);
}
} // namespace detail

template <class A, class S>
py::class_<bh::histogram<A, S>>
register_histogram(py::module &m, const char *name, const char *desc) {
    using histogram_t = bh::histogram<A, S>;
    using value_type  = typename histogram_t::value_type;
    namespace bv2     = boost::variant2;

    py::class_<histogram_t> hist(m, name, desc, py::buffer_protocol());

    hist.def(py::init<const A &, S>(), "axes"_a, "storage"_a = S())

        .def_buffer([](bh::histogram<A, S> &h) -> py::buffer_info {
            return make_buffer(h, false);
        })

        .def("rank", &histogram_t::rank, "Number of axes (dimensions) of histogram")
        .def("size",
             &histogram_t::size,
             "Total number of bins in the histogram (including underflow/overflow)")
        .def("reset", &histogram_t::reset, "Reset bin counters to zero")

        .def("__copy__", [](const histogram_t &self) { return histogram_t(self); })
        .def("__deepcopy__",
             [](const histogram_t &self, py::object memo) {
                 histogram_t *a  = new histogram_t(self);
                 py::module copy = py::module::import("copy");
                 for(unsigned i = 0; i < a->rank(); i++) {
                     bh::unsafe_access::axis(*a, i).metadata()
                         = copy.attr("deepcopy")(a->axis(i).metadata(), memo);
                 }
                 return a;
             })

        .def(py::self + py::self)

        .def(py::self == py::self)
        .def(py::self != py::self)

        .def_property_readonly_static(
            "_storage_type",
            [](py::object) {
                return py::detail::get_type_handle(
                    typeid(typename histogram_t::storage_type), true);
                // Change to py::type<T>() if added to PyBind11
            })

        ;

    // Atomics for example do not support these operations
    def_optionally(hist,
                   bh::detail::has_operator_rmul<histogram_t, double>{},
                   py::self *= double());
    def_optionally(hist,
                   bh::detail::has_operator_rmul<histogram_t, double>{},
                   py::self * double());
    def_optionally(hist,
                   bh::detail::has_operator_rmul<histogram_t, double>{},
                   double() * py::self);
    def_optionally(hist,
                   bh::detail::has_operator_rdiv<histogram_t, double>{},
                   py::self /= double());
    def_optionally(hist,
                   bh::detail::has_operator_rdiv<histogram_t, double>{},
                   py::self / double());

    hist.def(
            "to_numpy",
            [](histogram_t &h, bool flow) {
                py::tuple tup(1 + h.rank());

                // Add the histogram buffer as the first argument
                unchecked_set(tup, 0, py::array(make_buffer(h, flow)));

                // Add the axis edges
                h.for_each_axis([&tup, flow, i = 0u](const auto &ax) mutable {
                    unchecked_set(tup, ++i, axis::edges(ax, flow, true));
                });

                return tup;
            },
            "flow"_a = false)

        .def("_copy_in",
             [](histogram_t &h, py::array_t<double> input) { copy_in(h, input); })

        .def(
            "view",
            [](histogram_t &h, bool flow) { return py::array(make_buffer(h, flow)); },
            "flow"_a = false)

        .def(
            "axis",
            [](const histogram_t &self, int i) {
                unsigned ii = i < 0 ? self.rank() - (unsigned)std::abs(i) : (unsigned)i;
                if(ii < self.rank())
                    return self.axis(ii);
                else
                    throw std::out_of_range(
                        "The axis value must be less than the rank");
            },
            "i"_a,
            py::return_value_policy::move)

        .def(
            "at",
            [](const histogram_t &self, py::args &args) {
                auto int_args = py::cast<std::vector<int>>(args);
                return self.at(int_args);
            },
            "Select a contents given indices. Also consider [] indexing to get "
            "contents.")

        .def(
            "_at_set",
            [](histogram_t &self, const value_type &input, py::args &args) {
                auto int_args     = py::cast<std::vector<int>>(args);
                self.at(int_args) = input;
            },
            "Use [] indexing to set instead")

        .def("__repr__", &shift_to_string<histogram_t>)

        .def(
            "sum",
            [](const histogram_t &self, bool flow) {
                return sum_histogram(self, flow);
            },
            "flow"_a = false)

        .def(
            "empty",
            [](const histogram_t &self, bool flow) {
                return bh::algorithm::empty(
                    self, flow ? bh::coverage::all : bh::coverage::inner);
            },
            "flow"_a = false)

        .def("reduce",
             [](const histogram_t &self, py::args args) {
                 return bh::algorithm::reduce(
                     self, py::cast<std::vector<bh::algorithm::reduce_option>>(args));
             })

        .def("project",
             [](const histogram_t &self, py::args values) {
                 return bh::algorithm::project(self,
                                               py::cast<std::vector<unsigned>>(values));
             })

        .def("fill",
             [](histogram_t &self, py::args args, py::kwargs kwargs) {
                 using array_int_t
                     = py::array_t<int, py::array::c_style | py::array::forcecast>;
                 using array_double_t
                     = py::array_t<double, py::array::c_style | py::array::forcecast>;

                 if(args.size() != self.rank())
                     throw std::invalid_argument("Wrong number of args");

                 namespace bmp = boost::mp11;
                 static_assert(
                     bmp::mp_empty<bmp::mp_set_difference<
                         bmp::mp_unique<bmp::mp_transform<bh::axis::traits::value_type,
                                                          axis_variant>>,
                         bmp::mp_list<double, int, std::string>>>::value,
                     "supported value types are double, int, std::string; new axis was "
                     "added with different value type");

                 // HD: std::vector<std::string> is for passing strings, this very very
                 // inefficient but works at least I need to change something in
                 // boost::histogram to make passing strings from a numpy array
                 // efficient
                 using varg_t = boost::variant2::variant<array_int_t,
                                                         int,
                                                         array_double_t,
                                                         double,
                                                         std::vector<std::string>,
                                                         std::string>;
                 auto vargs   = bh::detail::make_stack_buffer<varg_t>(
                     bh::unsafe_access::axes(self));

                 {
                     auto args_it  = args.begin();
                     auto vargs_it = vargs.begin();
                     self.for_each_axis([&args_it, &vargs_it](const auto &ax) {
                         using T = std::decay_t<decltype(ax.value(0))>;
                         detail::set_varg(
                             boost::mp11::mp_identity<T>{}, *vargs_it++, *args_it++);
                     });
                 }

                 bool has_weight = false;
                 bv2::variant<array_double_t, double>
                     weight; // default constructed as empty array
                 {
                     auto w = optional_arg(kwargs, "weight");
                     if(!w.is_none()) {
                         has_weight = true;
                         if(detail::is_pyiterable(w))
                             weight = py::cast<array_double_t>(w);
                         else
                             weight = py::cast<double>(w);
                     }
                 }

                 using storage_t = typename histogram_t::storage_type;
                 bh::detail::static_if<detail::is_one_of<storage_t,
                                                         storage::mean,
                                                         storage::weighted_mean>>(
                     [&kwargs, &vargs, &weight, &has_weight](auto &h) {
                         auto s = required_arg(kwargs, "sample");
                         finalize_args(kwargs);

                         auto sarray = py::cast<array_double_t>(s);
                         if(sarray.ndim() != 1)
                             throw std::invalid_argument("Sample array must be 1D");

                         // HD: is it safe to release the gil? sarray is a Python
                         // object, could this cause trouble?
                         py::gil_scoped_release lock;
                         if(has_weight)
                             bv2::visit(
                                 [&h, &vargs, &sarray](const auto &w) {
                                     h.fill(vargs, bh::sample(sarray), bh::weight(w));
                                 },
                                 weight);
                         else
                             h.fill(vargs, bh::sample(sarray));
                     },
                     [&kwargs, &vargs, &weight, &has_weight](auto &h) {
                         finalize_args(kwargs);

                         py::gil_scoped_release lock;
                         if(has_weight)
                             bv2::visit(
                                 [&h, &vargs](const auto &w) {
                                     h.fill(vargs, bh::weight(w));
                                 },
                                 weight);
                         else
                             h.fill(vargs);
                     },
                     self);
                 return self;
             })

        .def(make_pickle<histogram_t>())

        ;

    return hist;
}
