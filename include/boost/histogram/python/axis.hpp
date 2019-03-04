// Copyright 2015-2018 Hans Dembinski and Henry Schreiner
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/histogram.hpp>

#include <array>
#include <vector>

namespace bh = boost::histogram;

using regular_axis = bh::axis::regular<>;

using regular_axes = bh::storage_adaptor<std::vector<regular_axis>>;
using regular_1D_axes = bh::storage_adaptor<std::array<regular_axis, 1>>;
using regular_2D_axes = bh::storage_adaptor<std::array<regular_axis, 2>>;
