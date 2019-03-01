// Copyright 2015-2018 Hans Dembinski and Henry Schreiner
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/histogram.hpp>

#include <vector>

namespace bh = boost::histogram;

using regular_axis = bh::axis::regular<>;
using regular_axes = std::vector<regular_axis>;
using regular_axes_storage = bh::storage_adaptor<regular_axes>;

using int_vector_storage = bh::storage_adaptor<std::vector<int>>;
