// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <boost/histogram/python/pybind11.hpp>

#include <boost/histogram/python/copyable_atomic.hpp>

#include <boost/histogram.hpp>

#include <boost/variant.hpp>

#include <cstdint>


namespace storage {

// Names match Python names
using int_ = bh::dense_storage<uint64_t>;
using atomic_int = bh::dense_storage<copyable_atomic<uint64_t>>;
using double_ = bh::dense_storage<double>;
using unlimited = bh::unlimited_storage<>;
using weight = bh::weight_storage;
using profile = bh::profile_storage;
using weighted_profile = bh::weighted_profile_storage;
    
// Some types not yet suppored (mostly due to fill not accepting weight and sample yet)
using any_variant = boost::variant<
    atomic_int,
    int_,
    double_,
    unlimited,
    weight
>;

}  // namespace storage
