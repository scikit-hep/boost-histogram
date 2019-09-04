// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <boost/histogram/python/pybind11.hpp>

#include <boost/histogram/accumulators/thread_safe.hpp>
#include <boost/histogram/storage_adaptor.hpp>
#include <boost/histogram/unlimited_storage.hpp>

#include <cstdint>

namespace storage {

// Names match Python names
using int_             = bh::dense_storage<uint64_t>;
using atomic_int       = bh::dense_storage<bh::accumulators::thread_safe<uint64_t>>;
using double_          = bh::dense_storage<double>;
using unlimited        = bh::unlimited_storage<>;
using weight           = bh::weight_storage;
using profile          = bh::profile_storage;
using weighted_profile = bh::weighted_profile_storage;

} // namespace storage
