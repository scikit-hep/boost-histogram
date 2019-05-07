// Copyright 2019 Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <boost/histogram/python/pybind11.hpp>

#include <boost/histogram/histogram.hpp>
#include <boost/histogram/detail/axes.hpp>
#include <boost/histogram/detail/meta.hpp>

#include <cassert>
#include <algorithm>
#include <stdexcept>

namespace impl {

template <class T>
struct py_array_type_impl {
  using type = T;
};

// specialization for category<std::string>
template <>
struct py_array_type_impl<std::string> {
  using type = const char*;
};

template <class A>
using py_array_type = typename py_array_type_impl<
    boost::histogram::detail::remove_cvref_t<
      boost::histogram::detail::arg_type<decltype(&A::index)>
    >
  >::type;
  
// convert all inputs to numpy arrays of the right axis type
template <class Axes>
std::size_t normalize_input(const Axes& axes, py::args args, py::object* values) {
  using namespace boost::histogram;
  unsigned i_axis = 0;
  std::size_t n_array = 0;
  detail::for_each_axis(axes, [&args, &values, &i_axis, &n_array](const auto& axis) {
    using A = detail::remove_cvref_t<decltype(axis)>;
    using T = py_array_type<A>;
    auto arr = py::cast<py::array_t<T>>(args[i_axis]);
    // allow arrays of dim 0 or dim == 1 with same length
    if (arr.ndim() == 1) {
      const auto n = arr.shape()[0];
      if (n_array != 0) {
        if (n_array != n)
          throw std::runtime_error("arrays must be scalars or have same length");
      }
      else {
        n_array = n;        
      }
    } else if (arr.ndim() > 1) {
      throw std::runtime_error("arrays must have dim 0 or 1");
    }
    values[i_axis] = arr;
    ++i_axis;
  });
  return n_array;
}

template <class Axes>
void fill_index_buffer(std::size_t offset, const std::size_t n,
                       Axes& axes, const py::object* values,
                       boost::histogram::axis::index_type* iter) {
  using namespace boost::histogram;
  // no support for growing axis yet
  assert(detail::has_growing_axis<Axes>::value == false);
  unsigned i_axis = 0;
  detail::for_each_axis(axes, [offset, n, iter, values, &i_axis](const auto& axis) {
    using A = detail::remove_cvref_t<decltype(axis)>;
    using T = py_array_type<A>;
    auto v = py::cast<py::array_t<T>>(values[i_axis]);
    if (v.ndim() == 1) {
      std::transform(v.data() + offset, v.data() + offset + n, iter,
                     [&axis](const T& x) { return axis.index(x); });
    } else {
      assert(v.ndim() == 0);
      std::fill(iter, iter + n, axis.index(v.data()[0]));
    }
    ++i_axis;
  });
}

// this should go to boostorg/histogram
template <class Axes>
void fill_strides(const Axes& axes, std::size_t* strides) {
  strides[0] = 1;
  boost::histogram::detail::for_each_axis(axes, [&strides](const auto& axis) {
    *++strides = *strides * boost::histogram::axis::traits::extent(axis);
  });
}
} // namespace impl

template <class Histogram>
void fill2(Histogram& h, py::args args, py::kwargs kwargs) {
  using namespace boost::histogram;

  const unsigned rank = h.rank();
  if (rank != args.size())
    throw std::invalid_argument("number of arguments must match histogram rank");

  auto& axes = unsafe_access::axes(h);
  auto values = detail::make_stack_buffer<py::object>(axes);
  const std::size_t n_array = impl::normalize_input(axes, args, values.data());

  constexpr std::size_t n_index = 1 << 14;
  boost::histogram::axis::index_type buffer[n_index];
  const std::size_t max_size = n_index / detail::get_size(axes);
  auto strides = detail::make_stack_buffer<std::size_t>(axes);
  impl::fill_strides(axes, strides.data());
  std::size_t i_array = 0;
  while (i_array != n_array) {
    const std::size_t n = std::min(max_size, n_array - i_array);
    impl::fill_index_buffer(i_array, n, axes, values.data(), buffer);
    // buffer is structured: a0:i0, ... , a0:iN, a1:i0, ... , a1:iN, ...
    auto& storage = unsafe_access::storage(h);
    for (std::size_t i = 0; i < n; ++i) {
      // calculate linear storage index manually
      std::size_t j = 0;
      auto bi = buffer + i;
      for (auto stride : strides) {
        j += stride * *bi;
        bi += n;
      }
      ++storage[j];
    }
    i_array += n;
  }
}

