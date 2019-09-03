// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <boost/core/ignore_unused.hpp>
#include <boost/histogram/python/kwargs.hpp>
#include <boost/histogram/python/pybind11.hpp>
#include <pybind11/operators.h>

#include <boost/histogram/python/axis.hpp>
#include <boost/histogram/python/histogram.hpp>
#include <boost/histogram/python/serializion.hpp>
#include <boost/histogram/python/storage.hpp>
#include <boost/histogram/python/typetools.hpp>

#include <boost/histogram.hpp>
#include <boost/histogram/algorithm/project.hpp>
#include <boost/histogram/algorithm/sum.hpp>
#include <boost/histogram/axis/ostream.hpp>
#include <boost/histogram/ostream.hpp>
#include <boost/histogram/unsafe_access.hpp>

#include <boost/mp11.hpp>

#include <future>
#include <sstream>
#include <thread>
#include <tuple>
#include <vector>

template <class A, class S>
py::class_<bh::histogram<A, S>> register_histogram(py::module &m, const char *name, const char *desc) {
    using histogram_t = bh::histogram<A, S>;

    py::class_<histogram_t> hist(m, name, desc, py::buffer_protocol());

    hist.def(py::init<const A &, S>(), "axes"_a, "storage"_a = S())

        .def_buffer([](bh::histogram<A, S> &h) -> py::buffer_info { return make_buffer(h, false); })

        .def("rank", &histogram_t::rank, "Number of axes (dimensions) of histogram")
        .def("size", &histogram_t::size, "Total number of bins in the histogram (including underflow/overflow)")
        .def("reset", &histogram_t::reset, "Reset bin counters to zero")

        .def("__copy__", [](const histogram_t &self) { return histogram_t(self); })
        .def("__deepcopy__",
             [](const histogram_t &self, py::object memo) {
                 histogram_t *a  = new histogram_t(self);
                 py::module copy = py::module::import("copy");
                 for(unsigned i = 0; i < a->rank(); i++) {
                     bh::unsafe_access::axis(*a, i).metadata() = copy.attr("deepcopy")(a->axis(i).metadata(), memo);
                 }
                 return a;
             })

        .def(py::self + py::self)

        .def(py::self == py::self)
        .def(py::self != py::self)

        ;

    // Atomics for example do not support these operations
    def_optionally(hist, bh::detail::has_operator_rmul<histogram_t, double>{}, py::self *= double());
    def_optionally(hist, bh::detail::has_operator_rmul<histogram_t, double>{}, py::self * double());
    def_optionally(hist, bh::detail::has_operator_rmul<histogram_t, double>{}, double() * py::self);
    def_optionally(hist, bh::detail::has_operator_rdiv<histogram_t, double>{}, py::self /= double());
    def_optionally(hist, bh::detail::has_operator_rdiv<histogram_t, double>{}, py::self / double());

    hist.def(
            "to_numpy",
            [](histogram_t &h, bool flow) {
                py::list listing;

                // Add the histogram as the first argument
                py::array arr(make_buffer(h, flow));
                listing.append(arr);

                // Add the axis edges
                for(unsigned i = 0; i < h.rank(); i++) {
                    const auto &ax = h.axis(i);
                    listing.append(axis_to_edges(ax, flow));
                }

                return py::cast<py::tuple>(listing);
            },
            "flow"_a = false,
            "convert to a numpy style tuple of returns")

        .def(
            "view",
            [](histogram_t &h, bool flow) { return py::array(make_buffer(h, flow)); },
            "flow"_a = false,
            "Return a view into the data, optionally with overflow turned on")

        .def(
            "axis",
            [](histogram_t &self, int i) {
                unsigned ii = i < 0 ? self.rank() - (unsigned)std::abs(i) : (unsigned)i;
                if(ii < self.rank())
                    return self.axis(ii);
                else
                    throw std::out_of_range("The axis value must be less than the rank");
            },
            "Get N-th axis with runtime index",
            "i"_a,
            py::return_value_policy::move)

        .def(
            "at",
            [](histogram_t &self, py::args &args) {
                // Optimize for no dynamic?
                auto int_args = py::cast<std::vector<int>>(args);
                return self.at(int_args);
            },
            "Access bin counter at indices")

        .def("__repr__", shift_to_string<histogram_t>())

        .def(
            "sum",
            [](const histogram_t &self, bool flow) {
                if(flow) {
                    return bh::algorithm::sum(self);
                } else {
                    using T       = typename bh::histogram<A, S>::value_type;
                    using AddType = boost::mp11::mp_if<std::is_arithmetic<T>, double, T>;
                    using Sum     = boost::mp11::mp_if<std::is_arithmetic<T>, bh::accumulators::sum<double>, T>;
                    Sum sum;
                    for(auto &&x : bh::indexed(self))
                        sum += (AddType)*x;
                    using R = boost::mp11::mp_if<std::is_arithmetic<T>, double, T>;
                    return static_cast<R>(sum);
                }
            },
            "flow"_a = false)

        /* Broken: Does not work if any string axes present (even just in variant) */
        .def(
            "rebin",
            [](const histogram_t &self, unsigned axis, unsigned merge) {
                return bh::algorithm::reduce(self, bh::algorithm::rebin(axis, merge));
            },
            "axis"_a,
            "merge"_a,
            "Rebin by merging bins. You must select an axis.")

        .def(
            "shrink",
            [](const histogram_t &self, unsigned axis, double lower, double upper) {
                return bh::algorithm::reduce(self, bh::algorithm::shrink(axis, lower, upper));
            },
            "axis"_a,
            "lower"_a,
            "upper"_a,
            "Shrink an axis. You must select an axis.")

        .def(
            "shrink_and_rebin",
            [](const histogram_t &self, unsigned axis, double lower, double upper, unsigned merge) {
                return bh::algorithm::reduce(self, bh::algorithm::shrink_and_rebin(axis, lower, upper, merge));
            },
            "axis"_a,
            "lower"_a,
            "upper"_a,
            "merge"_a,
            "Shrink an axis and rebin. You must select an axis.")

        .def(
            "project",
            [](const histogram_t &self, py::args values) {
                // If static
                // histogram<any_axis> any = self;
                return bh::algorithm::project(self, py::cast<std::vector<unsigned>>(values));
            },
            "Project to a single axis or several axes on a multidiminsional histogram")

        .def(
            "fill",
            [](histogram_t &self, py::args args, py::kwargs kwargs) {
                auto empty_weight = py::object();
                auto weight       = optional_arg(kwargs, "weight", empty_weight);

                using arrayd = py::array_t<double>;

                // TODO: compute this typelist from the value types of the supported axes
                using VArg = boost::variant2::variant<double, arrayd>;
                auto vargs = bh::detail::make_stack_buffer<VArg>(bh::unsafe_access::axes(self));

                if(args.size() != self.rank())
                    throw std::invalid_argument("Wrong number of args");

                unsigned iarg = 0;
                for(auto arg : args) {
                    if(py::isinstance<py::buffer>(arg) || py::hasattr(arg, "__iter__")) {
                        auto tmp       = py::cast<arrayd>(arg);
                        vargs.at(iarg) = tmp;
                        if(tmp.ndim() != 1)
                            throw std::invalid_argument("All arrays must be 1D");
                    } else {
                        vargs.at(iarg) = py::cast<double>(arg);
                    }
                    ++iarg;
                }

                // TODO handle weights
                using storage_t = typename histogram_t::storage_type;
                bh::detail::static_if<boost::mp11::mp_or<std::is_same<storage_t, storage::profile>,
                                                         std::is_same<storage_t, storage::weighted_profile>>>(
                    [&](auto &h) {
                        auto sample = required_arg<py::object>(kwargs, "sample");
                        finalize_args(kwargs);
                        auto sarray = py::cast<arrayd>(sample);
                        // HD: this causes an error in boost::histogram, needs to be fixed
                        // if(py::isinstance<double>(weight))
                        //     h.fill(vargs, bh::sample(sarray), bh::weight(py::cast<double>(weight)));
                        // else if(py::isinstance<arrayd>(weight))
                        //     h.fill(vargs, bh::sample(sarray), bh::weight(py::cast<arrayd>(weight)));
                        // else
                        h.fill(vargs, bh::sample(sarray));
                    },
                    [&](auto &h) {
                        finalize_args(kwargs);
                        if(py::isinstance<double>(weight))
                            h.fill(vargs, bh::weight(py::cast<double>(weight)));
                        else if(!weight.is(empty_weight) && py::isinstance<arrayd>(weight))
                            h.fill(vargs, bh::weight(py::cast<arrayd>(weight)));
                        else
                            h.fill(vargs);
                    },
                    self);
            },
            "Insert data into the histogram")

        .def(make_pickle<histogram_t>())

        ;

    return hist;
}
