// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <boost/histogram/python/pybind11.hpp>

#include <boost/histogram.hpp>
#include <cstdint>

namespace bh = boost::histogram;

using dense_int_storage = bh::dense_storage<uint64_t>;
using dense_double_storage = bh::dense_storage<double>;

using any_storage_variant = bh::axis::variant<
    dense_int_storage,
    dense_double_storage,
    bh::unlimited_storage<>,
    bh::weight_storage
>;

any_storage_variant extract_storage(py::kwargs kwargs);
