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

#include <boost/core/ignore_unused.hpp>
#include <boost/histogram/detail/accumulator_traits.hpp>
#include <boost/histogram/detail/axes.hpp>
#include <boost/histogram/sample.hpp>
#include <boost/histogram/unsafe_access.hpp>
#include <boost/histogram/weight.hpp>
#include <boost/mp11.hpp>
#include <boost/variant2/variant.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
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

    // NOLINTNEXTLINE(google-explicit-constructor)
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

inline bool fits_in_int(std::int64_t v) {
    return v >= static_cast<std::int64_t>(std::numeric_limits<int>::min())
           && v <= static_cast<std::int64_t>(std::numeric_limits<int>::max());
}

inline bool fits_in_int(std::uint64_t v) {
    return v <= static_cast<std::uint64_t>(std::numeric_limits<int>::max());
}

/// Narrow a wide or unsigned integer array to int, the value type of the integer
/// axes. NumPy would wrap out-of-range values silently.
template <class T>
inline c_array_t<int> narrow_to_int(py::handle x) {
    const py::array_t<T, py::array::c_style | py::array::forcecast> wide(
        py::reinterpret_borrow<py::object>(x));
    const std::vector<py::ssize_t> shape(wide.shape(), wide.shape() + wide.ndim());
    py::array_t<int> out(shape);

    const T* in = wide.data();
    int* op     = out.mutable_data();
    for(py::ssize_t i = 0, n = wide.size(); i < n; ++i) {
        if(!fits_in_int(in[i]))
            throw py::value_error(
                "Integer axis values must fit in a 32-bit signed integer");
        op[i] = static_cast<int>(in[i]);
    }

    return {py::reinterpret_borrow<py::object>(out)};
}

// Make sure float arrays don't get cast to integers (-.5 rounds to 0!)
template <>
inline decltype(auto) special_cast<c_array_t<int>>(py::handle x) {
    const auto dtype  = py::cast<py::array>(x).dtype();
    const char kind   = dtype.kind();
    const auto nbytes = dtype.itemsize();

    // These always fit in an int, so let NumPy convert them
    if(kind == 'b' || (kind == 'i' && nbytes <= 4) || (kind == 'u' && nbytes <= 2))
        return py::cast<c_array_t<int>>(x);
    if(kind == 'i')
        return narrow_to_int<std::int64_t>(x);
    if(kind == 'u')
        return narrow_to_int<std::uint64_t>(x);
    throw py::type_error("Only integer arrays supported when targeting integer axes");
}

// Produce a type error for passing float to int
template <>
inline decltype(auto) special_cast<int>(py::handle x) {
    try {
        return py::cast<int>(x);
    } catch(std::runtime_error&) {
        throw py::type_error(
            "Only integer values supported when targeting integer axes");
    }
}

// The first alternative must stay cheap to default construct; a stack buffer of
// these is made for every fill, and only the leading axes.size() are assigned.
using arg_t = variant::variant<double,
                               c_array_t<double>,
                               int,
                               c_array_t<int>,
                               std::string,
                               c_array_t<std::string>>;

using weight_t = variant::variant<variant::monostate, double, c_array_t<double>>;

// a sample is either a scalar (broadcast) or a 1D array
using sample_t = variant::variant<double, c_array_t<double>>;

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

inline auto get_sample(const py::handle& s) {
    sample_t sample;
    // a scalar sample is broadcast to match the other arguments, like a scalar
    // axis value or weight
    if(is_value<double>(s)) {
        sample = py::cast<double>(s);
    } else {
        auto sarray = py::cast<c_array_t<double>>(s);
        if(sarray.ndim() != 1)
            throw std::invalid_argument("Sample array must be 1D");
        sample = std::move(sarray);
    }
    return sample;
}

// Boost.Histogram < 1.91 reports sample types as const T&, newer versions as T
template <class T>
struct decayed_traits;

template <bool W, class... Ts>
struct decayed_traits<bh::detail::accumulator_traits_holder<W, Ts...>> {
    using type = bh::detail::accumulator_traits_holder<W, std::decay_t<Ts>...>;
};

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
    const py::gil_scoped_release lock;
    variant::visit(
        overload([&h, &vargs](const variant::monostate&) { h.fill(vargs); },
                 [&h, &vargs](const auto& w) { h.fill(vargs, bh::weight(w)); }),
        weight);
}

// for accumulators that accept a weight and a double
template <class Histogram, class VArgs>
void fill_impl(bh::detail::accumulator_traits_holder<true, double>,
               Histogram& h,
               const VArgs& vargs,
               const weight_t& weight,
               py::kwargs& kwargs) {
    auto s = required_arg(kwargs, "sample");
    finalize_args(kwargs);
    auto sample = get_sample(s);

    // releasing gil here is safe, we don't manipulate refcounts
    const py::gil_scoped_release lock;
    variant::visit(
        [&h, &vargs, &weight](const auto& sval) {
            variant::visit(overload(
                               [&h, &vargs, &sval](const variant::monostate&) {
                                   h.fill(vargs, bh::sample(sval));
                               },
                               [&h, &vargs, &sval](const auto& w) {
                                   h.fill(vargs, bh::sample(sval), bh::weight(w));
                               }),
                           weight);
        },
        sample);
}

// for multi_cell
template <class Histogram, class VArgs>
void fill_impl(bh::detail::accumulator_traits_holder<false, boost::span<double>>,
               Histogram& h,
               const VArgs& vargs,
               const weight_t& weight,
               py::kwargs& kwargs) {
    boost::ignore_unused(weight);
    auto s = required_arg(kwargs, "sample");
    finalize_args(kwargs);
    auto sarray = py::cast<c_array_t<double>>(s);
    if(sarray.ndim() != 2)
        throw std::invalid_argument("Sample or weight array for MultiCell must be 2D");

    auto buf              = sarray.request();
    const auto buf_shape0 = static_cast<std::size_t>(buf.shape[0]);
    const auto buf_shape1 = static_cast<std::size_t>(buf.shape[1]);

    // A row that is not nelem wide would throw deep inside the fill loop
    const auto nelem = bh::unsafe_access::storage(h).nelem();
    if(buf_shape1 != nelem)
        throw std::invalid_argument("Sample or weight array for MultiCell must have "
                                    + std::to_string(nelem) + " entries per row, got "
                                    + std::to_string(buf_shape1));

    // releasing gil here is safe, we don't manipulate refcounts
    const py::gil_scoped_release lock;
    auto* src = static_cast<double*>(buf.ptr);
    std::vector<boost::span<double>> vec_s;
    vec_s.reserve(buf_shape0);
    for(std::size_t i = 0; i < buf_shape0; i++) {
        vec_s.emplace_back(src + (i * buf_shape1), buf_shape1);
    }
    h.fill(vargs, bh::sample(vec_s));
}

} // namespace detail

template <class Histogram>
Histogram& fill(Histogram& self, const py::args& args, py::kwargs kwargs) {
    using value_type = typename Histogram::value_type;
    using traits     = typename detail::decayed_traits<
        bh::detail::accumulator_traits<value_type>>::type;
    detail::fill_impl(traits{},
                      self,
                      detail::get_vargs(bh::unsafe_access::axes(self), args),
                      detail::get_weight(kwargs),
                      kwargs);
    return self;
}
