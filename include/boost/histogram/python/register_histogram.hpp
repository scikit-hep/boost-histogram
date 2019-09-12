// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <boost/histogram/python/kwargs.hpp>
#include <boost/histogram/python/pybind11.hpp>

#include <boost/histogram/algorithm/project.hpp>
#include <boost/histogram/algorithm/reduce.hpp>
#include <boost/histogram/algorithm/sum.hpp>
#include <boost/histogram/axis/ostream.hpp>
#include <boost/histogram/histogram.hpp>
#include <boost/histogram/ostream.hpp>
#include <boost/histogram/python/axis.hpp>
#include <boost/histogram/python/histogram.hpp>
#include <boost/histogram/python/indexed.hpp>
#include <boost/histogram/python/kwargs.hpp>
#include <boost/histogram/python/serializion.hpp>
#include <boost/histogram/python/shared_histogram.hpp>
#include <boost/histogram/python/storage.hpp>
#include <boost/histogram/python/variant.hpp>
#include <boost/histogram/python/typetools.hpp>
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

// this or something similar should move to boost::histogram::axis::traits
template <class Axis>
using get_axis_value_type = boost::histogram::python::remove_cvref_t<decltype(std::declval<Axis>().value(0))>;

template <class T>
bool is_pyiterable(const T &t) {
    return py::isinstance<py::buffer>(t) || py::hasattr(t, "__iter__");
}

template <class T, class VArg, class Arg>
void set_varg(boost::mp11::mp_identity<T>, VArg &v, const Arg &x) {
    if(is_pyiterable(x)) {
        auto arr = py::cast<py::array_t<T>>(x);
        if(arr.ndim() != 1)
            throw std::invalid_argument("All arrays must be 1D");
        v = arr;
    } else
        v = py::cast<T>(x);
}

// specialization for string (HD: this is very inefficient and will be made more efficient in the future)
template <class VArg, class Arg>
void set_varg(boost::mp11::mp_identity<std::string>, VArg &v, const Arg &x) {
    if(py::isinstance<py::str>(x))
        v = py::cast<std::string>(x);
    else
        v = py::cast<std::vector<std::string>>(x);
}
} // namespace detail

template <class A, class S>
py::class_<bh::histogram<A, S>> register_histogram(py::module &m, const char *name, const char *desc) {
    using histogram_t = bh::histogram<A, S>;
    namespace bv2     = boost::variant2;

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
            [](const histogram_t &self, int i) {
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
            [](const histogram_t &self, py::args &args) {
                // Optimize for no dynamic?
                auto int_args = py::cast<std::vector<int>>(args);
                return self.at(int_args);
            },
            "Access bin counter at indices")

        .def("__repr__", shift_to_string<histogram_t>())

        .def(
            "__getitem__",
            [](const histogram_t &self,
               py::object index) -> bv2::variant<typename histogram_t::value_type, histogram_t> {
                // If this is not a tuple (>1D), make it a tuple of 1D
                // Then, convert tuples to list
                py::list indexes;
                if(py::isinstance<py::tuple>(index))
                    indexes = py::cast<py::tuple>(index);
                else
                    indexes.append(index);

                // Expand ... to :
                indexes = expand_ellipsis(indexes, self.rank());

                if(indexes.size() != self.rank())
                    throw std::out_of_range("You must provide the same number of indices as the rank of the histogram");

                // Allow [bh.loc(...)] to work
                for(py::size_t i = 0; i < indexes.size(); i++) {
                    if(py::hasattr(indexes[i], "value"))
                        indexes[i]
                            = self.axis(static_cast<unsigned>(i)).index(py::cast<double>(indexes[i].attr("value")));
                }

                // If this is (now) all integers, return the bin contents
                try {
                    auto int_args = py::cast<std::vector<int>>(indexes);
                    return self.at(int_args);
                } catch(const py::cast_error &) {
                }

                // Compute needed slices and projections
                std::vector<bh::algorithm::reduce_option> slices;
                std::vector<unsigned> projections;
                std::tie(slices, projections) = get_slices(
                    indexes,
                    [&self](bh::axis::index_type i, double val) {
                        return self.axis(static_cast<unsigned>(i)).index(val);
                    },
                    [&self](bh::axis::index_type i) { return self.axis(static_cast<unsigned>(i)).size(); });

                if(projections.empty())
                    return bh::algorithm::reduce(self, slices);
                else {
                    auto reduced = bh::algorithm::reduce(self, slices);
                    return bh::algorithm::project(self, projections);
                }
            })

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

        .def(
            "reduce",
            [](const histogram_t &self, py::args args) {
                return bh::algorithm::reduce(self, py::cast<std::vector<bh::algorithm::reduce_option>>(args));
            },
            "Reduce based on one or more reduce_option")

        .def(
            "project",
            [](const histogram_t &self, py::args values) {
                return bh::algorithm::project(self, py::cast<std::vector<unsigned>>(values));
            },
            "Project to a single axis or several axes on a multidiminsional histogram")

        .def(
            "fill",
            [](histogram_t &self, py::args args, py::kwargs kwargs) {
                if(args.size() != self.rank())
                    throw std::invalid_argument("Wrong number of args");

                namespace bmp = boost::mp11;
                static_assert(
                    bmp::mp_empty<bmp::mp_set_difference<
                        bmp::mp_unique<bmp::mp_transform<::detail::get_axis_value_type, bmp::mp_first<axes::any>>>,
                        bmp::mp_list<double, int, std::string>>>::value,
                    "supported value types are double, int, std::string; new axis was added with different value type");

                // HD: std::vector<std::string> is for passing strings, this very very inefficient but works at least
                // I need to change something in boost::histogram to make passing strings from a numpy array efficient
                using varg_t = boost::variant2::
                    variant<py::array_t<double>, double, py::array_t<int>, int, std::vector<std::string>, std::string>;
                auto vargs = bh::detail::make_stack_buffer<varg_t>(bh::unsafe_access::axes(self));

                {
                    auto args_it  = args.begin();
                    auto vargs_it = vargs.begin();
                    self.for_each_axis([&args_it, &vargs_it](const auto &ax) {
                        using T = typename bh::python::remove_cvref_t<decltype(ax.value(0))>;
                        detail::set_varg(boost::mp11::mp_identity<T>{}, *vargs_it++, *args_it++);
                    });
                }

                bool has_weight = false;
                bv2::variant<py::array_t<double>, double> weight; // default constructed as empty array
                {
                    auto w = optional_arg(kwargs, "weight", py::none());
                    if(!w.is_none()) {
                        has_weight = true;
                        if(detail::is_pyiterable(w))
                            weight = py::cast<py::array_t<double>>(w);
                        else
                            weight = py::cast<double>(w);
                    }
                }

                using storage_t = typename histogram_t::storage_type;
                bh::detail::static_if<detail::is_one_of<storage_t, storage::profile, storage::weighted_profile>>(
                    [&kwargs, &vargs, &weight, &has_weight](auto &h) {
                        auto s = required_arg<py::object>(kwargs, "sample");
                        finalize_args(kwargs);

                        auto sarray = py::cast<py::array_t<double>>(s);
                        if(sarray.ndim() != 1)
                            throw std::invalid_argument("Sample array must be 1D");

                        // HD: is it safe to release the gil? sarray is a Python object, could this cause trouble?
                        py::gil_scoped_release lock;
                        if(has_weight)
                            bv2::visit([&h, &vargs, &sarray](
                                           const auto &w) { h.fill(vargs, bh::sample(sarray), bh::weight(w)); },
                                       weight);
                        else
                            h.fill(vargs, bh::sample(sarray));
                    },
                    [&kwargs, &vargs, &weight, &has_weight](auto &h) {
                        finalize_args(kwargs);

                        py::gil_scoped_release lock;
                        if(has_weight)
                            bv2::visit([&h, &vargs](const auto &w) { h.fill(vargs, bh::weight(w)); }, weight);
                        else
                            h.fill(vargs);
                    },
                    self);
            },
            "Insert data into the histogram")

        .def(make_pickle<histogram_t>())

        ;

    hist.def(
        "indexed",
        [](histogram_t &self, bool flow) {
            return make_repeatable_iterator(self, flow ? bh::coverage::all : bh::coverage::inner);
        },
        "flow"_a = false,
        "Set up an iterator, returns a special accessor for bin info and content",
        py::keep_alive<0, 1>());

    register_ufunc_tools(hist);

    register_indexed<histogram_t>(m, name);

    return hist;
}
