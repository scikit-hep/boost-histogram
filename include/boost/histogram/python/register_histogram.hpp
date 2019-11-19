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
#include <boost/histogram/ostream.hpp>
#include <boost/histogram/python/accumulators/ostream.hpp>
#include <boost/histogram/python/axis.hpp>
#include <boost/histogram/python/histogram.hpp>
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

template <class...>
struct overload_t;

template <class F>
struct overload_t<F> : F {
    overload_t(F &&f)
        : F(std::forward<F>(f)) {}
    using F::operator();
};

template <class F, class... Fs>
struct overload_t<F, Fs...> : F, overload_t<Fs...> {
    overload_t(F &&x, Fs &&... xs)
        : F(std::forward<F>(x))
        , overload_t<Fs...>(std::forward<Fs>(xs)...) {}
    using F::operator();
    using overload_t<Fs...>::operator();
};

template <class... Fs>
auto overload(Fs &&... xs) {
    return overload_t<Fs...>(std::forward<Fs>(xs)...);
}

template <class T>
struct c_array_t : py::array_t<T, py::array::c_style | py::array::forcecast> {
    using base_t = py::array_t<T, py::array::c_style | py::array::forcecast>;
    using base_t::base_t;
    std::size_t size() const { return static_cast<std::size_t>(base_t::size()); }
};

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

        .def("rank", &histogram_t::rank)
        .def("size", &histogram_t::size)
        .def("reset", &histogram_t::reset)

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
            [](py::object self, bool flow) {
                auto &h   = py::cast<histogram_t &>(self);
                auto info = make_buffer(h, flow);
                return py::array(
                    pybind11::dtype(info), info.shape, info.strides, info.ptr, self);
                // Note that, due to the missing signature py::array(info, self), we
                // have to write this out fully here. TODO: Make PR to PyBind11
            },
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

        .def("at",
             [](const histogram_t &self, py::args &args) -> value_type {
                 auto int_args = py::cast<std::vector<int>>(args);
                 return self.at(int_args);
             })

        .def("_at_set",
             [](histogram_t &self, const value_type &input, py::args &args) {
                 auto int_args     = py::cast<std::vector<int>>(args);
                 self.at(int_args) = input;
             })

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
                 if(args.size() != self.rank())
                     throw std::invalid_argument("Wrong number of args");

                 using detail::is_pyiterable;
                 using detail::is_one_of;
                 using detail::overload;
                 using detail::c_array_t;

                 namespace bmp = boost::mp11;
                 static_assert(
                     bmp::mp_empty<bmp::mp_set_difference<
                         bmp::mp_unique<bmp::mp_transform<bh::axis::traits::value_type,
                                                          axis_variant>>,
                         bmp::mp_list<double, int, std::string>>>::value,
                     "supported value types are double, int, std::string; new axis was "
                     "added with different value type");

                 // HD: std::vector<std::string> is for passing strings, this is very
                 // inefficient but works at least. I need to change something in
                 // boost::histogram to make passing strings from a numpy array
                 // efficient.
                 using varg_t = boost::variant2::variant<c_array_t<double>,
                                                         double,
                                                         c_array_t<int>,
                                                         int,
                                                         std::vector<std::string>,
                                                         std::string>;

                 auto vargs = bh::detail::make_stack_buffer<varg_t>(
                     bh::unsafe_access::axes(self));

                 self.for_each_axis([args_it  = args.begin(),
                                     vargs_it = vargs.begin()](const auto &ax) mutable {
                     using T = bh::axis::traits::value_type<std::decay_t<decltype(ax)>>;

                     bh::detail::static_if<std::is_same<T, std::string>>(
                         [](auto, varg_t &v, const py::handle &x) {
                             // specialization for string (HD: this is very inefficient
                             // and will be made more efficient in the future)
                             if(py::isinstance<py::str>(x))
                                 // hot-fix, should be `v = py::cast<std::string>(x);`
                                 // once the issue in boost::histogram is fixed
                                 v = std::vector<std::string>(1,
                                                              py::cast<std::string>(x));
                             else if(py::isinstance<py::array>(x)) {
                                 if(py::cast<py::array>(x).ndim() != 1)
                                     throw std::invalid_argument(
                                         "All arrays must be 1D");
                                 v = py::cast<std::vector<std::string>>(x);
                             } else {
                                 v = py::cast<std::vector<std::string>>(x);
                             }
                         },
                         [](auto u, varg_t &v, const py::handle &x) {
                             using U = typename decltype(u)::type;
                             if(is_pyiterable(x)) {
                                 auto arr = py::cast<c_array_t<U>>(x);
                                 if(arr.ndim() != 1)
                                     throw std::invalid_argument(
                                         "All arrays must be 1D");
                                 v = arr;
                             } else
                                 v = py::cast<U>(x);
                         },
                         bmp::mp_identity<T>(),
                         *vargs_it++,
                         *args_it++);
                 });

                 // default constructed as monostate to indicate absence of weight
                 bv2::variant<bv2::monostate, double, c_array_t<double>> weight;
                 {
                     auto w = optional_arg(kwargs, "weight");
                     if(!w.is_none()) {
                         if(is_pyiterable(w))
                             weight = py::cast<c_array_t<double>>(w);
                         else
                             weight = py::cast<double>(w);
                     }
                 }

                 using storage_t = typename histogram_t::storage_type;
                 bh::detail::static_if<
                     is_one_of<storage_t, storage::mean, storage::weighted_mean>>(
                     [&kwargs, &vargs, &weight](auto &h) {
                         auto s = required_arg(kwargs, "sample");
                         finalize_args(kwargs);

                         auto sarray = py::cast<c_array_t<double>>(s);
                         if(sarray.ndim() != 1)
                             throw std::invalid_argument("Sample array must be 1D");

                         // releasing gil here is safe, we don't manipulate refcounts
                         py::gil_scoped_release lock;
                         bv2::visit(
                             overload(
                                 [&h, &vargs, &sarray](const bv2::monostate &) {
                                     h.fill(vargs, bh::sample(sarray));
                                 },
                                 [&h, &vargs, &sarray](const auto &w) {
                                     h.fill(vargs, bh::sample(sarray), bh::weight(w));
                                 }),
                             weight);
                     },
                     [&kwargs, &vargs, &weight](auto &h) {
                         finalize_args(kwargs);

                         // releasing gil here is safe, we don't manipulate refcounts
                         py::gil_scoped_release lock;
                         bv2::visit(
                             overload([&h, &vargs](
                                          const bv2::monostate &) { h.fill(vargs); },
                                      [&h, &vargs](const auto &w) {
                                          h.fill(vargs, bh::weight(w));
                                      }),
                             weight);
                     },
                     self);
                 return self;
             })

        .def("_reset_row",
             [](histogram_t &self, unsigned ax, int row) {
                 // Reset just a single row. Used by indexing as a workaround
                 // to remove the flow bins when missing crop

                 for(auto &&ind : bh::indexed(self, bh::coverage::all)) {
                     if(ind.index(ax) == row) {
                         *ind = typename histogram_t::value_type();
                     }
                 }
             })

        .def(make_pickle<histogram_t>())

        ;

    return hist;
}
