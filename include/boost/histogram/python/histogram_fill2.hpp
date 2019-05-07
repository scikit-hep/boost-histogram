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

namespace detail {
// convert all inputs to numpy arrays of the right axis type
template <class Axes, class Values>
std::size_t normalize_input(const Axes& axes, py::args args, Values& values) {
  using namespace boost::histogram;
  unsigned i_axis = 0;
  std::size_t n_array = 0;
  detail::for_each_axis([&args, &values, &i_axis, &n_array](const auto& axis) mutable {
    using T = detail::arg_type<detail::remove_cv_t<decltype(axis)>>;
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
  }, axes);
  return n_array;
}

template <class Axes, class Values, class Buffer>
void fill_index_buffer(std::size_t offset, const std::size_t n,
                       Axes& axes, const Values& values, BufferIterator iter) {
  using namespace boost::histogram;
  assert(detail::has_growing_axis<Axes>::value == false, "no support for growing axis yet");
  unsigned i_axis = 0;
  detail::for_each_axis([&iter, &values, &i_axis, offset, n](auto& axis) mutable {
    using T = detail::arg_type<detail::remove_cv_t<decltype(axis)>>;
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

template <class Axes>
void fill_strides(const Axes& axes, std::size_t* strides) {
  strides[0] = 1;
  detail::for_each_axis([&strides](const auto& axis) {
    *++strides = *strides * axis::traits::extent(axis);
  }, axes);
}
} // namespace detail

template <class Histogram>
void fill2(Histogram& h, py::args args, py::kwargs kwargs) {
  using namespace boost::histogram;

  const unsigned rank = h.rank();
  if (rank != args.size())
    throw std::invalid_argument("number of arguments must match histogram rank");

  auto& axes = unsafe_access::axes(h);
  auto values = detail::make_stack_buffer<py::object>(axes);
  const std::size_t n_array = detail::normalize_input(axes, args, values);

  constexpr std::size_t n_index = 1 << 16;
  axis::index_type buffer[n_index];
  const std::size_t max_size = n_index / detail::get_size(axes);
  auto strides = detail::make_stack_buffer<std::size_t>(axes);
  detail::fill_strides(axes, strides);
  while (i_array < n_array) {
    const std::size_t n = std::min(max_size, n_array - i_array);
    detail::fill_index_buffer(i_array, n, axes, values, n_array, buffer);
    // buffer is structured: a0:i0, ... , a0:iN, a1:i0, ... , a1:iN, ...
    auto& storage = unsafe_access::storage(h);
    for (std::size_t i = 0; i < n; ++i) {
      // calculate linear storage index manually
      std::size_t j = 0;
      auto iter = buffer + i;
      for (auto stride : strides) {
        j += stride * *iter;
        iter += n_iter;
      }
      ++storage.at(j);
    }
  }
}

