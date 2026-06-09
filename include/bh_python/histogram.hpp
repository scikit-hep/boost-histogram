// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <bh_python/pybind11.hpp>

#include <bh_python/accumulators/mean.hpp>
#include <bh_python/accumulators/weighted_mean.hpp>
#include <bh_python/accumulators/weighted_sum.hpp>
#include <bh_python/axis.hpp>
#include <bh_python/multi_cell.hpp>
#include <bh_python/storage.hpp>

#include <boost/histogram/algorithm/empty.hpp>
#include <boost/histogram/algorithm/sum.hpp>
#include <boost/histogram/detail/axes.hpp>
#include <boost/histogram/histogram.hpp>
#include <boost/histogram/unsafe_access.hpp>

#include <unordered_map>
#include <vector>

namespace pybind11 {

/// The descriptor for atomic_* is the same as the descriptor for *, as long this uses
/// standard layout
template <class T>
struct format_descriptor<bh::accumulators::count<T, true>> : format_descriptor<T> {
    static_assert(std::is_standard_layout<bh::accumulators::count<T, true>>::value, "");
};

} // namespace pybind11

namespace detail {

template <class Axes, class T>
py::buffer_info make_buffer_impl(const Axes& axes, bool flow, T* ptr) {
    // strides are in bytes
    auto shape         = bh::detail::make_stack_buffer<py::ssize_t>(axes);
    auto strides       = bh::detail::make_stack_buffer<py::ssize_t>(axes);
    py::ssize_t stride = sizeof(T);
    unsigned rank      = 0;
    char* start        = reinterpret_cast<char*>(ptr);
    bh::detail::for_each_axis(axes, [&](const auto& axis) {
        const bool underflow
            = bh::axis::traits::options(axis) & bh::axis::option::underflow;
        if(!flow && underflow)
            start += stride;
        const auto extent = bh::axis::traits::extent(axis);
        shape[rank]       = flow ? extent : axis.size();
        strides[rank]     = stride;
        stride *= extent;
        ++rank;
    });

    return py::buffer_info(
        start,                              // Pointer to buffer
        sizeof(T),                          // Size of one scalar
        py::format_descriptor<T>::format(), // Python format descriptor
        static_cast<py::ssize_t>(rank),     // Number of dimensions
        shape,                              // Buffer shape
        strides                             // Strides (in bytes) for each index
    );
}

struct double_converter {
    template <class T, class Buffer>
    void operator()(T* tp, Buffer& b) const {
        b.template make<double>(b.size, tp);
    }

    template <class Buffer>
    void operator()(double*, Buffer&) const {} // nothing to do
};
} // namespace detail

/// Build and return a buffer over the current data.
/// Flow controls whether under/over flow bins are present
template <class A, class T>
py::buffer_info make_buffer(bh::histogram<A, bh::dense_storage<T>>& h, bool flow) {
    const auto& axes = bh::unsafe_access::axes(h);
    auto& storage    = bh::unsafe_access::storage(h);
    return detail::make_buffer_impl(axes, flow, &storage[0]);
}

/// Sparse storage keeps only filled cells in a hash map, so there is no
/// contiguous buffer to expose. Callers should use ``to_coo()`` instead. We
/// throw rather than return an empty buffer so ``.view()`` / ``np.asarray`` give
/// a single, clear error.
template <class A, class T>
py::buffer_info make_buffer(bh::histogram<A, storage::sparse_storage<T>>& /* h */,
                            bool /* flow */) {
    throw py::type_error(
        "sparse storage does not support .view() or buffer access; use .to_coo()");
}

/// Specialization for unlimited_buffer
template <class A, class Allocator>
py::buffer_info make_buffer(bh::histogram<A, bh::unlimited_storage<Allocator>>& h,
                            bool flow) {
    const auto& axes = bh::unsafe_access::axes(h);
    auto& storage    = bh::unsafe_access::storage(h);
    // User requested a view into the memory of unlimited storage. We convert
    // the internal storage to double now to avoid the view becoming invalid
    // upon changes to the histogram. This is the only way to provide a safe
    // view, because we cannot automatically update the view when the
    // underlying memory buffer changes. In practice it is ok, because users
    // usually want to get the view after filling the histogram, and then the
    // counts are usually converted to doubles for further processing anyway.
    auto& buffer = bh::unsafe_access::unlimited_storage_buffer(storage);
    buffer.visit(detail::double_converter(), buffer);
    return detail::make_buffer_impl(axes, flow, static_cast<double*>(buffer.ptr));
}

/// From C++17
template <class T>
constexpr std::add_const_t<T>& std_as_const(T& t) noexcept {
    return t;
}

/// Specialization for multi_cell buffer
template <class A, class T>
py::buffer_info make_buffer(bh::histogram<A, bh::multi_cell<T>>& h, bool flow) {
    const auto& axes = bh::unsafe_access::axes(h);
    auto& storage    = bh::unsafe_access::storage(h);
    using AxesVec    = std::decay_t<decltype(axes)>;
    AxesVec new_axes;
    // Add the cells as a first pseudo-axis to treat them correctly for the buffer
    // creation. This will create a buffer in the shape (nelem, axis_1, axis_2, ...)
    // where nelem is the number of cells per bin This also coincides with how the
    // cells are stored on the multi cell storage side Having the cells as the
    // last dimension might feel more natural, but does not work with the current
    // storage implementation
    new_axes.emplace_back(axis::integer_none{0, static_cast<int>(storage.nelem())});
    new_axes.insert(std::end(new_axes), std::begin(axes), std::end(axes));
    return detail::make_buffer_impl(
        std_as_const(new_axes), flow, static_cast<double*>(storage.get_buffer()));
}

/// Trait identifying our sparse (hash-map backed) storages.
template <class S>
struct is_sparse_storage : std::false_type {};

template <class T>
struct is_sparse_storage<storage::sparse_storage<T>> : std::true_type {};

namespace detail {
/// Per-axis layout used to (un)ravel a flat sparse storage index. The flat index
/// counts flow bins, with underflow (when present) at position 0, exactly like
/// make_buffer_impl above.
struct axis_layout {
    std::size_t extent;    // size + has_underflow + has_overflow
    std::size_t size;      // number of normal (in-range) bins
    std::size_t underflow; // 1 if this axis has an underflow bin, else 0
};

template <class Axes>
std::vector<axis_layout> make_axis_layout(const Axes& axes) {
    std::vector<axis_layout> layout;
    bh::detail::for_each_axis(axes, [&layout](const auto& ax) {
        const bool has_underflow
            = bh::axis::traits::options(ax) & bh::axis::option::underflow;
        layout.push_back({static_cast<std::size_t>(bh::axis::traits::extent(ax)),
                          static_cast<std::size_t>(ax.size()),
                          has_underflow ? std::size_t{1} : std::size_t{0}});
    });
    return layout;
}

/// True if a flat sparse storage key addresses an in-range (non-flow) cell.
inline bool is_inner_cell(std::size_t lin, const std::vector<axis_layout>& layout) {
    for(const auto& info : layout) {
        const std::size_t iflow = lin % info.extent;
        lin /= info.extent;
        if(iflow < info.underflow || iflow >= info.underflow + info.size)
            return false;
    }
    return true;
}
} // namespace detail

/// Extract the filled cells of a sparse histogram in COO form. Returns a tuple
/// ``(indices, values)`` where ``indices`` is an ``(ndim, n)`` integer array and
/// ``values`` an ``n``-length array. With ``flow=False`` the flow cells are
/// dropped and indices run 0..size-1; with ``flow=True`` indices run
/// 0..extent-1 (underflow at 0), matching the ``view(flow=True)`` grid.
template <class A, class T>
py::tuple histogram_to_coo(bh::histogram<A, storage::sparse_storage<T>>& h, bool flow) {
    const auto& axes  = bh::unsafe_access::axes(h);
    auto& storage     = bh::unsafe_access::storage(h);
    const auto layout = detail::make_axis_layout(axes);
    const auto rank   = layout.size();

    // storage_adaptor publicly inherits map_impl<T>, which publicly inherits the
    // underlying map, so this upcast exposes only the filled cells.
    using map_type  = std::unordered_map<std::size_t, T>;
    const auto& map = static_cast<const map_type&>(storage);

    std::vector<std::vector<py::ssize_t>> idx_cols(rank);
    std::vector<T> vals;
    vals.reserve(map.size());

    std::vector<py::ssize_t> multi(rank);
    for(const auto& kv : map) {
        std::size_t lin = kv.first;
        bool keep       = true;
        for(std::size_t ax = 0; ax < rank; ++ax) {
            const auto& info        = layout[ax];
            const std::size_t iflow = lin % info.extent;
            lin /= info.extent;
            if(flow) {
                multi[ax] = static_cast<py::ssize_t>(iflow);
            } else {
                const py::ssize_t real = static_cast<py::ssize_t>(iflow)
                                         - static_cast<py::ssize_t>(info.underflow);
                if(real < 0 || real >= static_cast<py::ssize_t>(info.size)) {
                    keep = false;
                    break;
                }
                multi[ax] = real;
            }
        }
        if(!keep)
            continue;
        for(std::size_t ax = 0; ax < rank; ++ax)
            idx_cols[ax].push_back(multi[ax]);
        vals.push_back(kv.second);
    }

    const auto n = static_cast<py::ssize_t>(vals.size());
    py::array_t<py::ssize_t> indices({static_cast<py::ssize_t>(rank), n});
    if(n > 0) {
        auto ind = indices.template mutable_unchecked<2>();
        for(std::size_t ax = 0; ax < rank; ++ax)
            for(py::ssize_t j = 0; j < n; ++j)
                ind(static_cast<py::ssize_t>(ax), j)
                    = idx_cols[ax][static_cast<std::size_t>(j)];
    }

    py::array_t<T> values(n);
    std::copy(vals.begin(), vals.end(), values.mutable_data());

    return py::make_tuple(std::move(indices), std::move(values));
}

/// Inverse of histogram_to_coo: scatter ``(indices, values)`` back into a sparse
/// histogram. Uses the same flow convention as histogram_to_coo. This is the
/// only bulk write path for sparse storage, since the numpy-buffer based
/// __setitem__ cannot densify the map.
template <class A, class T>
void histogram_from_coo(bh::histogram<A, storage::sparse_storage<T>>& h,
                        const py::array_t<py::ssize_t>& indices,
                        const py::array_t<T>& values,
                        bool flow) {
    const auto& axes  = bh::unsafe_access::axes(h);
    const auto layout = detail::make_axis_layout(axes);
    const auto rank   = layout.size();

    auto ind = indices.template unchecked<2>();
    auto val = values.template unchecked<1>();
    if(static_cast<std::size_t>(ind.shape(0)) != rank)
        throw py::value_error("indices first dimension must equal histogram rank");

    const auto n = val.shape(0);
    if(ind.shape(1) != n)
        throw py::value_error("indices second dimension must match values length");
    std::vector<int> at_index(rank);
    for(py::ssize_t j = 0; j < n; ++j) {
        for(std::size_t ax = 0; ax < rank; ++ax) {
            const py::ssize_t coord = ind(static_cast<py::ssize_t>(ax), j);
            // Convert back to boost's at() signed index (underflow -1, overflow size).
            const py::ssize_t signed_idx
                = flow ? coord - static_cast<py::ssize_t>(layout[ax].underflow) : coord;
            at_index[ax] = static_cast<int>(signed_idx);
        }
        h.at(at_index) = val(j);
    }
}

template <class A, class S>
auto histogram_sum(const bh::histogram<A, S>& h, bool flow) {
    // A rank-0 histogram has no flow bins, so inner == all. Use all to avoid
    // Boost's indexed range, which is UB for rank-0 (it reads uninitialized
    // per-axis state); the all path uses plain iteration instead.
    const auto cov = (flow || h.rank() == 0) ? bh::coverage::all : bh::coverage::inner;
    return bh::algorithm::sum(h, cov);
}

/// The storage iterator of a map-backed histogram visits every *logical* cell
/// (hash lookup per cell), so bh::algorithm::sum would be O(dense size) — for
/// the huge axis spaces sparse storage targets, that never finishes. Sum only
/// the filled cells instead; flow=false unravels each key to skip flow cells.
template <class A, class T>
T histogram_sum(const bh::histogram<A, storage::sparse_storage<T>>& h, bool flow) {
    using map_type  = std::unordered_map<std::size_t, T>;
    const auto& map = static_cast<const map_type&>(bh::unsafe_access::storage(h));

    T total{};
    if(flow) {
        for(const auto& kv : map)
            total += kv.second;
        return total;
    }

    const auto layout = detail::make_axis_layout(bh::unsafe_access::axes(h));
    for(const auto& kv : map)
        if(detail::is_inner_cell(kv.first, layout))
            total += kv.second;
    return total;
}

template <class A, class S>
bool histogram_empty(const bh::histogram<A, S>& h, bool flow) {
    if(h.rank() == 0) {
        // algorithm::empty drives the same rank-0-UB indexed range; check the
        // single cell directly instead.
        using value_type = typename bh::histogram<A, S>::value_type;
        return !(*h.begin() != value_type());
    }
    return bh::algorithm::empty(h, flow ? bh::coverage::all : bh::coverage::inner);
}

/// Sparse counterpart of bh::algorithm::empty, iterating only the filled cells
/// (see histogram_sum). Explicit zeros (left behind by e.g. -=) count as empty.
template <class A, class T>
bool histogram_empty(const bh::histogram<A, storage::sparse_storage<T>>& h, bool flow) {
    using map_type  = std::unordered_map<std::size_t, T>;
    const auto& map = static_cast<const map_type&>(bh::unsafe_access::storage(h));

    if(flow) {
        for(const auto& kv : map)
            if(kv.second != T{})
                return false;
        return true;
    }

    const auto layout = detail::make_axis_layout(bh::unsafe_access::axes(h));
    for(const auto& kv : map)
        if(kv.second != T{} && detail::is_inner_cell(kv.first, layout))
            return false;
    return true;
}

template <class A, class S>
bool histogram_equal(const bh::histogram<A, S>& a, const bh::histogram<A, S>& b) {
    return a == b;
}

/// Sparse counterpart of histogram::operator==, whose storage comparison walks
/// every logical cell (see histogram_sum). Compare the two maps directly,
/// treating absent keys as zero — the maps are not structurally identical for
/// equal histograms, since -= can leave explicit zeros behind.
template <class A, class T>
bool histogram_equal(const bh::histogram<A, storage::sparse_storage<T>>& a,
                     const bh::histogram<A, storage::sparse_storage<T>>& b) {
    if(!bh::detail::axes_equal(bh::unsafe_access::axes(a), bh::unsafe_access::axes(b)))
        return false;

    using map_type = std::unordered_map<std::size_t, T>;
    const auto& ma = static_cast<const map_type&>(bh::unsafe_access::storage(a));
    const auto& mb = static_cast<const map_type&>(bh::unsafe_access::storage(b));

    for(const auto& kv : ma) {
        const auto it = mb.find(kv.first);
        if(kv.second != (it == mb.end() ? T{} : it->second))
            return false;
    }
    for(const auto& kv : mb)
        if(kv.second != T{} && ma.find(kv.first) == ma.end())
            return false;
    return true;
}
