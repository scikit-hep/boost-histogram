// Copyright 2019 Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <boost/histogram/python/pybind11.hpp>
#include <boost/histogram/python/kwargs.hpp>

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

namespace bh = boost::histogram;

namespace detail {

template <class Axes, class F>
void for_each_axis_impl(std::true_type, Axes &&axes, F &&f) {
    for(auto &&x : axes)
        bh::axis::visit(std::forward<F>(f), x);
}

template <class Axes, class F>
void for_each_axis_impl(std::false_type, Axes &&axes, F &&f) {
    for(auto &&x : axes)
        std::forward<F>(f)(x);
}

template <class Axes, class F>
void for_each_axis(Axes &&axes, F &&f) {
    using U = boost::mp11::mp_first<std::decay_t<Axes>>;
    for_each_axis_impl(bh::detail::is_axis_variant<U>(), axes, std::forward<F>(f));
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
using py_array_type =
    typename py_array_type_impl<bh::detail::remove_cvref_t<bh::detail::arg_type<decltype(&A::index)>>>::type;

template <class T>
std::pair<const T *, bool> normalize_array(std::size_t &size, py::array_t<T> &t) {
    // allow arrays of dim 0 or dim == 1 with same length
    switch(t.ndim()) {
    case 0:
        break;
    case 1:
        if(size == 0)
            size = static_cast<std::size_t>(t.shape(0));
        else if(size != static_cast<std::size_t>(t.shape(0)))
            throw std::invalid_argument("arrays must be broadcastable to same length");
        break;
    default:
        throw std::invalid_argument("arrays must have dim 0 or 1");
    }
    return std::make_pair(t.data(), t.ndim() == 0);
}

struct normalize_result {
    const std::size_t size; // length of input
    const bool scalar_weight, scalar_sample;
    const double *weight_ptr;
    const double *sample_ptr;
};

// - converts all inputs to numpy arrays of the right axis type
// - casts weights and samples to double; could be refined to allow more types
//   at the cost of more complex code
template <class Axes>
normalize_result
normalize_input(const Axes &axes, py::args args, py::object *values, py::object &weight, py::object &sample) {
    unsigned iax     = 0;
    std::size_t size = 0;
    for_each_axis(axes, [&args, &values, &iax, &size](const auto &axis) {
        using A = bh::detail::remove_cvref_t<decltype(axis)>;
        using T = py_array_type<A>;
        auto a  = py::cast<py::array_t<T>>(args[iax]);
        normalize_array(size, a);
        values[iax] = a;
        ++iax;
    });

    std::pair<const double *, bool> w, s;

    if(!weight.is_none()) {
        auto a = py::cast<py::array_t<double>>(weight);
        w      = normalize_array(size, a);
    }

    if(!sample.is_none()) {
        auto a = py::cast<py::array_t<double>>(sample);
        s      = normalize_array(size, a);
    }

    // if all arguments are scalars, set size to 1
    return normalize_result{std::max(size, static_cast<std::size_t>(1)), w.second, s.second, w.first, s.first};
}

// this class can be extended to also handle weights and samples
template <class Axis, class Storage, class T>
struct fill_1d_helper {
    using axis_index_type = boost::histogram::axis::index_type;
    Axis &axis;
    Storage &storage;

    void operator()(const T &t) const {
        using Opt = bh::axis::traits::static_options<Axis>;
        impl1(Opt::test(bh::axis::option::growth), t);
    }

    void impl1(std::false_type, const T &t) const {
        using Opt = bh::axis::traits::static_options<Axis>;

        const auto i = axis.index(t);
        impl2(Opt::test(bh::axis::option::underflow), Opt::test(bh::axis::option::overflow), i);
    }

    void impl1(std::true_type, const T &t) const {
        using Opt = bh::axis::traits::static_options<Axis>;

        const auto i_s = axis.update(t);
        if(i_s.second != 0)
            boost::histogram::detail::grow_storage(std::forward_as_tuple(axis), storage, &i_s.second);

        impl2(Opt::test(bh::axis::option::underflow), Opt::test(bh::axis::option::overflow), i_s.first);
    }

    void impl2(std::false_type, std::false_type, const axis_index_type i) const {
        if(0 <= i && i < axis.size())
            ++storage[static_cast<std::size_t>(i)];
    }

    void impl2(std::false_type, std::true_type, const axis_index_type i) const {
        if(0 <= i)
            ++storage[static_cast<std::size_t>(i)];
    }

    void impl2(std::true_type, std::false_type, const axis_index_type i) const {
        if(i < axis.size())
            ++storage[static_cast<std::size_t>(i + 1)];
    }

    void impl2(std::true_type, std::true_type, const axis_index_type i) const {
        ++storage[static_cast<std::size_t>(i + 1)];
    }
};

template <class Axes, class Storage>
void fill_1d(Axes &axes, Storage &storage, const py::object values, const normalize_result &nr) {
    if(nr.weight_ptr)
        throw std::runtime_error("weights not yet supported");
    if(nr.sample_ptr)
        throw std::runtime_error("samples not yet supported");

    for_each_axis(axes, [n = nr.size, &storage, values](auto &&axis) {
        using A = bh::detail::remove_cvref_t<decltype(axis)>;
        using T = detail::py_array_type<A>;

        const auto v = py::cast<py::array_t<T>>(values);
        assert(v.ndim() < 2); // precondition: ndim either 0 or 1 after normalizing
        const T *tp = v.data();

        std::for_each(tp, tp + n, fill_1d_helper<A, Storage, T>{axis, storage});
    });
}

static constexpr auto invalid_index = std::numeric_limits<std::size_t>::max();

// std::size_t-like integer with a persistent invalid state, similar to NaN
struct index_with_invalid_state {
    bool is_valid() const { return value != invalid_index; }

    index_with_invalid_state &operator=(const std::size_t x) {
        value = x;
        return *this;
    }

    index_with_invalid_state &operator+=(const std::size_t x) {
        if(is_valid()) {
            if(x == invalid_index)
                value = invalid_index;
            else
                value += x;
        }
        return *this;
    }

    index_with_invalid_state &operator++() {
        if(is_valid())
            ++value;
        return *this;
    }

    index_with_invalid_state &operator*(const std::size_t x) {
        if(is_valid())
            value *= x;
        return *this;
    }

    std::size_t operator-(const index_with_invalid_state &x) const {
        assert(value >= x.value);
        return (is_valid() && x.is_valid()) ? static_cast<std::size_t>(value - x.value) : invalid_index;
    }

    std::size_t value;
};

template <class Axis, class T>
struct fill_indices_helper {
    using axis_index_type = boost::histogram::axis::index_type;
    Axis &axis;
    const std::size_t stride;
    using index_iterator = index_with_invalid_state *;
    index_iterator begin, iter;

    fill_indices_helper(Axis &a, std::size_t s, index_iterator i)
        : axis(a)
        , stride(s)
        , begin(i)
        , iter(i) {}

    void maybe_shift_previous_indices(index_iterator iter, axis_index_type shift) {
        if(shift > 0)
            while(iter != begin)
                *--iter += static_cast<std::size_t>(shift) * stride;
    }

    void operator()(const T &t) {
        using Opt = bh::axis::traits::static_options<Axis>;
        impl1(Opt::test(bh::axis::option::growth), t);
    }

    void impl1(std::false_type, const T &t) {
        using Opt    = bh::axis::traits::static_options<Axis>;
        const auto i = axis.index(t);
        impl2(Opt::test(bh::axis::option::underflow), Opt::test(bh::axis::option::overflow), i);
    }

    void impl1(std::true_type, const T &t) {
        using Opt      = bh::axis::traits::static_options<Axis>;
        const auto i_s = axis.update(t);
        maybe_shift_previous_indices(iter, i_s.second);
        impl2(Opt::test(bh::axis::option::underflow), Opt::test(bh::axis::option::overflow), i_s.first);
    }

    void impl2(std::false_type, std::false_type, const axis_index_type i) {
        *iter++ += (0 <= i && i < axis.size()) ? static_cast<std::size_t>(i) * stride : invalid_index;
    }

    void impl2(std::false_type, std::true_type, const axis_index_type i) {
        *iter++ += 0 <= i ? static_cast<std::size_t>(i) * stride : invalid_index;
    }

    void impl2(std::true_type, std::false_type, const axis_index_type i) {
        *iter++ += i < axis.size() ? static_cast<std::size_t>(i + 1) * stride : invalid_index;
    }

    void impl2(std::true_type, std::true_type, const axis_index_type i) {
        *iter++ += static_cast<std::size_t>(i + 1) * stride;
    }
};

template <class Axes>
void fill_indices(
    index_with_invalid_state *indices, std::size_t offset, const std::size_t n, Axes &axes, const py::object *values) {
    std::fill(indices, indices + n, 0); // initialize to zero

    std::size_t stride = 1;
    for_each_axis(axes, [offset, n, &values, indices, &stride](auto &&axis) {
        using A = bh::detail::remove_cvref_t<decltype(axis)>;
        using T = py_array_type<A>;

        const auto v = py::cast<py::array_t<T>>(*values++);
        assert(v.ndim() < 2); // precondition: ndim either 0 or 1 after normalizing
        const T *tp = v.data();
        if(v.ndim() == 0) {
            // only one value to compute
            const auto old_value = *indices;
            fill_indices_helper<A, T>{axis, stride, indices}(*tp);
            const auto new_value    = *indices;
            const std::size_t shift = new_value - old_value;
            std::for_each(indices + 1, indices + n, [shift](auto &x) { x += shift; });
        } else {
            tp += offset;
            std::for_each(tp, tp + n, fill_indices_helper<A, T>{axis, stride, indices});
        }

        stride *= static_cast<std::size_t>(bh::axis::traits::extent(axis));
    });
}

template <class Storage>
struct storage_filler {
    Storage &storage;
    const std::size_t offset;
    normalize_result nr; // intentionally a copy

    using T = typename Storage::value_type;

    void operator()(const index_with_invalid_state &j) { impl1(bh::detail::has_operator_preincrement<T>{}, j); }

    void impl1(std::false_type, const index_with_invalid_state &j) {
        using Args = bh::detail::args_type<decltype(&T::operator())>;
        impl2(boost::mp11::mp_size<Args>{}, j);
    }

    void impl2(boost::mp11::mp_size_t<0>, const index_with_invalid_state &j) {
        if(nr.sample_ptr)
            throw std::invalid_argument("accumulator doesn't accept samples");
        if(nr.weight_ptr) {
            if(j.is_valid())
                storage[j.value](*nr.weight_ptr);
            if(!nr.scalar_weight)
                ++nr.weight_ptr;
        } else {
            if(j.is_valid())
                storage[j.value]();
        }
    }

    void impl1(std::true_type, const index_with_invalid_state &j) {
        if(nr.weight_ptr) {
            if(j.is_valid())
                storage[j.value] += *nr.weight_ptr;
            if(!nr.scalar_weight)
                ++nr.weight_ptr;
        } else {
            if(j.is_valid())
                ++storage[j.value];
        }
    }
};
} // namespace detail

template <class Histogram>
void fill2(Histogram &h, py::args args, py::kwargs kwargs) {
    const unsigned rank = h.rank();
    if(rank != args.size())
        throw std::invalid_argument("number of arguments must match histogram rank");

    auto &axes    = bh::unsafe_access::axes(h);
    auto &storage = bh::unsafe_access::storage(h);

    py::object weight = optional_arg(kwargs, "weight", py::none());
    py::object sample = optional_arg(kwargs, "sample", py::none());

    auto values   = bh::detail::make_stack_buffer<py::object>(axes);
    const auto nr = detail::normalize_input(axes, args, values.data(), weight, sample);

    if(rank == 1) {
        // run faster implementation for 1D which doesn't need an index buffer
        // note: specialization for rank==2 could also be added
        detail::fill_1d(axes, storage, values[0], nr);
    } else {
        constexpr std::size_t buffer_size = 1 << 14;
        detail::index_with_invalid_state indices[buffer_size];

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
        for(std::size_t offset = 0; offset < nr.size; offset += buffer_size) {
            const std::size_t n = std::min(buffer_size, nr.size - offset);
            // fill buffer of indices...
            detail::fill_indices(indices, offset, n, axes, values.data());
            // ...and fill corresponding storage cells
            std::for_each(
                indices, indices + n, detail::storage_filler<typename Histogram::storage_type>{storage, offset, nr});
        }
    }
}
