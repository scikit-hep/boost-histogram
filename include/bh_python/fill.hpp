// Copyright 2018-2019 Hans Dembinski and Henry Schreiner
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <bh_python/pybind11.hpp>

#include <bh_python/axis.hpp>
#include <bh_python/kwargs.hpp>
#include <bh_python/overload.hpp>
#include <bh_python/vector_string_caster.hpp>

#include <boost/histogram/detail/accumulator_traits.hpp>
#include <boost/histogram/detail/axes.hpp>
#include <boost/histogram/sample.hpp>
#include <boost/histogram/unsafe_access.hpp>
#include <boost/histogram/weight.hpp>
#include <boost/mp11.hpp>
#include <boost/variant2/variant.hpp>

#include <stdexcept>
#include <type_traits>
#include <vector>

namespace detail {

namespace mp11    = boost::mp11;
namespace variant = boost::variant2;

static_assert(
    mp11::mp_empty<mp11::mp_set_difference<
        mp11::mp_unique<mp11::mp_transform<bh::axis::traits::value_type, axis_variant>>,
        mp11::mp_list<double, int, std::string>>>::value,
    "supported value types are double, int, std::string; "
    "new axis was added with different value type");

template <class T>
struct c_array_t : py::array_t<T, py::array::c_style | py::array::forcecast> {
    using base_t = py::array_t<T, py::array::c_style | py::array::forcecast>;
    using base_t::base_t;
    using base_t::operator=;
    std::size_t size() const { return static_cast<std::size_t>(base_t::size()); }
};

// not actually a numpy array
template <>
struct c_array_t<std::string> : std::vector<std::string> {
    using base_t = std::vector<std::string>;
    using base_t::base_t;
    using base_t::operator=;

    c_array_t(base_t&& x)
        : base_t(std::move(x)) {}

    c_array_t& operator=(base_t&& x) {
        base_t::operator=(std::move(x));
        return *this;
    }
};

// for int, double
template <class T>
bool is_value(py::handle h) {
    if(py::isinstance<py::array>(h) && py::cast<py::array>(h).ndim() > 0)
        return false;
    return PyNumber_Check(h.ptr());
}

// for std::string
template <>
inline bool is_value<std::string>(py::handle h) {
    return py::isinstance<py::str>(h)
           || (py::isinstance<py::array>(h) && py::cast<py::array>(h).ndim() == 0);
}

template <class T>
decltype(auto) special_cast(py::handle x) {
    return py::cast<T>(x);
}

// allow conversion of dim 0 arrays
template <>
inline decltype(auto) special_cast<std::string>(py::handle x) {
    if(py::isinstance<py::array>(x))
        return py::cast<std::string>(py::cast<py::str>(x));
    return py::cast<std::string>(x);
}

// easier than specializing type_caster for c_array_t<std::string>
template <>
inline decltype(auto) special_cast<c_array_t<std::string>>(py::handle x) {
    using B = typename c_array_t<std::string>::base_t;
    return py::cast<B>(x);
}

using arg_t = variant::variant<c_array_t<double>,
                               double,
                               c_array_t<int>,
                               int,
                               c_array_t<std::string>,
                               std::string>;

using weight_t = variant::variant<variant::monostate, double, c_array_t<double>>;

inline auto get_vargs(const vector_axis_variant& axes, const py::args& args) {
    if(args.size() != axes.size())
        throw std::invalid_argument("Wrong number of args");

    auto vargs = bh::detail::make_stack_buffer<arg_t>(axes);

    bh::detail::for_each_axis(
        axes,
        [args_it = args.begin(), vargs_it = vargs.begin()](const auto& ax) mutable {
            using T = bh::axis::traits::value_type<std::decay_t<decltype(ax)>>;
            // T is one of: int, double, std::string

            const auto& x = *args_it++;
            auto& v       = *vargs_it++;

            if(is_value<T>(x)) {
                v = special_cast<T>(x);
            } else {
                if(py::isinstance<py::array>(x) && py::cast<py::array>(x).ndim() != 1)
                    throw std::invalid_argument("All arrays must be 1D");
                v = special_cast<c_array_t<T>>(x);
            }
        });

    return vargs;
}

inline auto get_weight(py::kwargs& kwargs) {
    // default constructed as monostate to indicate absence of weight
    variant::variant<variant::monostate, double, c_array_t<double>> weight;
    auto w = optional_arg(kwargs, "weight");
    if(!w.is_none()) {
        if(is_value<double>(w))
            weight = py::cast<double>(w);
        else
            weight = py::cast<c_array_t<double>>(w);
    }
    return weight;
}

// for accumulators that accept a weight
template <class Histogram, class VArgs>
void fill_impl(bh::detail::accumulator_traits_holder<true>,
               Histogram& h,
               const VArgs& vargs,
               const weight_t& weight,
               py::kwargs& kwargs) {
    none_only_arg(kwargs, "sample");
    finalize_args(kwargs);

    // releasing gil here is safe, we don't manipulate refcounts
    py::gil_scoped_release lock;
    variant::visit(
        overload([&h, &vargs](const variant::monostate&) { h.fill(vargs); },
                 [&h, &vargs](const auto& w) { h.fill(vargs, bh::weight(w)); }),
        weight);
}

// for accumulators that accept a weight and a double
template <class Histogram, class VArgs>
void fill_impl(bh::detail::accumulator_traits_holder<true, const double&>,
               Histogram& h,
               const VArgs& vargs,
               const weight_t& weight,
               py::kwargs& kwargs) {
    auto s = required_arg(kwargs, "sample");
    finalize_args(kwargs);
    auto sarray = py::cast<c_array_t<double>>(s);

    if(sarray.ndim() != 1)
        throw std::invalid_argument("Sample array must be 1D");

    // releasing gil here is safe, we don't manipulate refcounts
    py::gil_scoped_release lock;
    variant::visit(
        overload([&h, &vargs, &sarray](
                     const variant::monostate&) { h.fill(vargs, bh::sample(sarray)); },
                 [&h, &vargs, &sarray](const auto& w) {
                     h.fill(vargs, bh::sample(sarray), bh::weight(w));
                 }),
        weight);
}

} // namespace detail

template <class Histogram>
Histogram& fill(Histogram& self, py::args args, py::kwargs kwargs) {
    using value_type = typename Histogram::value_type;
    detail::fill_impl(bh::detail::accumulator_traits<value_type>{},
                      self,
                      detail::get_vargs(bh::unsafe_access::axes(self), args),
                      detail::get_weight(kwargs),
                      kwargs);
    return self;
}
