#pragma once

#include <boost/histogram.hpp>
#include <vector>

namespace bh = boost::histogram;

using vector_int_storage = bh::storage_adaptor<std::vector<int>>;
