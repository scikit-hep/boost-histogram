#pragma once

#include <boost/histogram.hpp>
#include <vector>
#include <cstdint>

namespace bh = boost::histogram;

using vector_int_storage = bh::storage_adaptor<std::vector<uint64_t>>;
