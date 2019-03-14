// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <boost/histogram.hpp>
#include <cstdint>

namespace bh = boost::histogram;

using dense_int_storage = bh::dense_storage<uint64_t>;
using dense_double_storage = bh::dense_storage<double>;
