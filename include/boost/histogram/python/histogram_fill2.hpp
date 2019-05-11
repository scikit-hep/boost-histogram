// Copyright 2019 Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <boost/histogram/python/pybind11.hpp>

#include <boost/histogram/histogram.hpp>
#include <boost/histogram/detail/axes.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/axis/traits.hpp>
#include <boost/histogram/axis/option.hpp>
#include <boost/histogram/axis/variant.hpp>
#include <boost/mp11.hpp>

#include <cassert>
#include <algorithm>
#include <stdexcept>
#include <type_traits>

namespace detail {

// all for_each_axis* should be replaced once the functionality is implemented in boost::histogram

template <class Axes, class F>
void for_each_axis_impl(std::true_type, Axes &&axes, F &&f) {
    for(auto &&x : axes)
        boost::histogram::axis::visit(std::forward<F>(f), x);
}

template <class Axes, class F>
void for_each_axis_impl(std::false_type, Axes &&axes, F &&f) {
    for(auto &&x : axes)
        std::forward<F>(f)(x);
}

template <class Axes, class F>
void for_each_axis(Axes &&axes, F &&f) {
    using U = boost::mp11::mp_first<std::decay_t<Axes>>;
    for_each_axis_impl(boost::histogram::detail::is_axis_variant<U>(), axes, std::forward<F>(f));
}

template <class... Ts, class F>
void for_each_axis(const std::tuple<Ts...> &axes, F &&f) {
    boost::mp11::tuple_for_each(axes, std::forward<F>(f));
}

template <class... Ts, class F>
void for_each_axis(std::tuple<Ts...> &axes, F &&f) {
    boost::mp11::tuple_for_each(axes, std::forward<F>(f));
}

template <class T>
struct py_array_type_impl {
    using type = T;
};

// specialization for category<std::string>
template <>
struct py_array_type_impl<std::string> {
    using type = const char *;
};

template <class A>
using py_array_type = typename py_array_type_impl<
    boost::histogram::detail::remove_cvref_t<boost::histogram::detail::arg_type<decltype(&A::index)>>>::type;

// convert all inputs to numpy arrays of the right axis type
template <class Axes>
std::size_t normalize_input(const Axes &axes, py::args args, py::object *values) {
    using namespace boost::histogram;
    unsigned i_axis     = 0;
    std::size_t n_array = 0;
    for_each_axis(axes, [&args, &values, &i_axis, &n_array](const auto &axis) {
        using A = bh::detail::remove_cvref_t<decltype(axis)>;
        using T = py_array_type<A>;
        auto x  = py::cast<py::array_t<T>>(args[i_axis]);
        // allow arrays of dim 0 or dim == 1 with same length
        if(x.ndim() == 1) {
            const auto n = static_cast<unsigned>(x.shape(0));
            if(n_array != 0) {
                if(n_array != n)
                    throw std::runtime_error("arrays must be scalars or have same length");
            } else {
                n_array = n;
            }
        } else if(x.ndim() > 1) {
            throw std::runtime_error("arrays must have dim 0 or 1");
        }
        values[i_axis++] = x;
    });
    // if all arguments are scalars, return 1
    return std::max(n_array, static_cast<std::size_t>(1));
}

template <class Axis, class Storage, class T>
struct fill_1d_helper {
    Axis &axis;
    Storage &storage;
    void operator()(const T &t) const {
        using Opt = bh::axis::traits::static_options<Axis>;
        impl(Opt::test(bh::axis::option::underflow),
             Opt::test(bh::axis::option::overflow),
             Opt::test(bh::axis::option::growth),
             t);
    }

    void impl(std::false_type, std::false_type, std::false_type, const T &t) const {
        const auto i = axis.index(t);
        if(0 <= i && i < axis.size())
            ++storage[static_cast<std::size_t>(i)];
    }

    void impl(std::false_type, std::true_type, std::false_type, const T &t) const {
        const auto i = axis.index(t);
        if(0 <= i)
            ++storage[static_cast<std::size_t>(i)];
    }

    void impl(std::true_type, std::false_type, std::false_type, const T &t) const {
        const auto i = axis.index(t);
        if(i < axis.size())
            ++storage[static_cast<std::size_t>(i + 1)];
    }

    void impl(std::true_type, std::true_type, std::false_type, const T &t) const {
        const auto i = axis.index(t);
        ++storage[static_cast<std::size_t>(i + 1)];
    }

    void impl(std::false_type, std::false_type, std::true_type, const T &t) const {
        const auto i_s = axis.update(t);
        if(i_s.second != 0)
            boost::histogram::detail::grow_storage(std::forward_as_tuple(axis), storage, &i_s.second);
        if(0 <= i_s.first && i_s.first < axis.size())
            ++storage[static_cast<std::size_t>(i_s.first)];
    }

    void impl(std::false_type, std::true_type, std::true_type, const T &t) const {
        const auto i_s = axis.update(t);
        if(i_s.second != 0)
            boost::histogram::detail::grow_storage(std::forward_as_tuple(axis), storage, &i_s.second);
        if(0 <= i_s.first)
            ++storage[static_cast<std::size_t>(i_s.first)];
    }

    void impl(std::true_type, std::false_type, std::true_type, const T &t) const {
        const auto i_s = axis.update(t);
        if(i_s.second != 0)
            boost::histogram::detail::grow_storage(std::forward_as_tuple(axis), storage, &i_s.second);
        if(i_s.first < axis.size())
            ++storage[static_cast<std::size_t>(i_s.first + 1)];
    }

    void impl(std::true_type, std::true_type, std::true_type, const T &t) const {
        const auto i_s = axis.update(t);
        if(i_s.second != 0)
            boost::histogram::detail::grow_storage(std::forward_as_tuple(axis), storage, &i_s.second);
        ++storage[static_cast<std::size_t>(i_s.first + 1)];
    }
};

template <class Axes, class Storage>
void fill_1d(const std::size_t n, Axes &axes, Storage &storage, const py::object values) {
    namespace bh = boost::histogram;
    for_each_axis(axes, [n, &storage, values](auto &&axis) {
        using A = bh::detail::remove_cvref_t<decltype(axis)>;
        using T = detail::py_array_type<A>;

        const auto v = py::cast<py::array_t<T>>(values);
        assert(v.ndim() < 2); // precondition: ndim either 0 or 1 after normalizing
        const T *tp = v.data();

        std::for_each(tp, tp + n, fill_1d_helper<A, Storage, T>{axis, storage});
    });
}

template <class Axes>
void fill_indices(std::size_t offset, const std::size_t n, Axes &axes, const py::object *values, std::size_t *iter) {
    namespace bh = boost::histogram;

    unsigned i_axis    = 0;
    std::size_t stride = 1;
    for_each_axis(axes, [offset, n, values, &iter, &i_axis, &stride](const auto &axis) {
        using A = bh::detail::remove_cvref_t<decltype(axis)>;
        using T = py_array_type<A>;

        constexpr auto opt = bh::axis::traits::static_options<A>{};
        if(opt & bh::axis::option::growth)
            throw std::runtime_error("no support for growing axis yet");
        constexpr auto shift = opt & bh::axis::option::underflow ? 1 : 0;

        const auto v = py::cast<py::array_t<T>>(values[i_axis]);
        assert(v.ndim() < 2); // precondition: ndim either 0 or 1 after normalizing
        const T *tp = v.data() + offset;
        if(v.ndim() == 1) {
            std::for_each(tp, tp + n, [&axis, shift, stride, &iter](const T &t) {
                const auto i = axis.index(t) + shift;
                if(i >= 0)
                    *iter += static_cast<std::size_t>(i) * stride;
                else
                    *iter = 0;
                ++iter;
            });
        } else {
            const auto i = axis.index(*tp) + shift;
            if(i >= 0)
                std::for_each(
                    iter, iter + n, [i, stride](std::size_t &j) { j += static_cast<std::size_t>(i) * stride; });
            else
                std::fill(iter, iter + n, 0);
        }

        stride *= static_cast<std::size_t>(bh::axis::traits::extent(axis));
        iter += n;
        ++i_axis;
    });
}
} // namespace detail

template <class Histogram>
void fill2(Histogram &h, py::args args, py::kwargs /* kwargs */) {
    namespace bh = boost::histogram;

    const unsigned rank = h.rank();
    if(rank != args.size())
        throw std::invalid_argument("number of arguments must match histogram rank");

    auto &axes    = bh::unsafe_access::axes(h);
    auto &storage = bh::unsafe_access::storage(h);

    auto values               = bh::detail::make_stack_buffer<py::object>(axes);
    const std::size_t n_array = detail::normalize_input(axes, args, values.data());

    if(rank == 1) {
        // run faster implementation for 1D which doesn't need an index buffer
        // note: specialization for rank==2 could also be added
        detail::fill_1d(n_array, axes, storage, values[0]);
    } else {
        constexpr std::size_t buffer_size = 1 << 14;
        std::size_t indices[buffer_size];

        /*
          Parallelization options for generic case.

          A) Run the whole fill2 method in parallel, each thread fills its own buffer of
          indices, synchronization (atomics) are needed to synchronize the incrementing of
          the storage cells. This leads to a lot of congestion for small histograms.

          B) Run only detail::fill_indices in parallel, subsections of the indices buffer
          can be filled by different threads. The final loop that fills the storage runs
          in the main thread, this requires no synchronization for the storage, cells do
          not need to support atomic operations.

          C) Like B), then sort the indices in the main thread and fill the
          storage in parallel, where each thread uses a disjunct set of indices. This
          should create less congestion and requires no synchronization for the storage.

          Note on C): Let's say we have an axis with 5 bins (with *flow to simplify).
          Then after filling 10 values, converting to indices and sorting, the index
          buffer may look like this: 0 0 0 1 2 2 2 4 4 5. Let's use two threads to fill
          the storage. Still in the main thread, we compute an iterator to the middle of
          the index buffer and move it to the right until the pointee changes. Now we have
          two ranges which contain disjunct sets of indices. We pass these ranges to the
          threads which then fill the storage. Since the threads by construction do not
          compete to increment the same cell, no further synchronization is required.

          In all cases, growing axes cannot be parallelized.
        */
        std::size_t i_array = 0;
        while(i_array != n_array) {
            const std::size_t n = std::min(buffer_size, n_array - i_array);
            // fill buffer of indices...
            std::fill(indices, indices + n, 1); // initialize
            detail::fill_indices(i_array, n, axes, values.data(), indices);
            // ...and increment corresponding storage cells
            std::for_each(indices, indices + n, [&storage](const std::size_t &j) {
                if(j > 0)
                    ++storage[static_cast<std::size_t>(j - 1)];
            });
            i_array += n;
        }
    }
}
