// Copyright 2018 Henry Schreiner
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <boost/histogram.hpp>
#include <vector>
#include <cstdint>

namespace bh = boost::histogram;

using vector_int_storage = bh::storage_adaptor<std::vector<uint64_t>>;
