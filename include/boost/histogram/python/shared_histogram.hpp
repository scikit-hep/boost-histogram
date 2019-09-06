// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <boost/histogram/python/pybind11.hpp>

#include <boost/histogram/algorithm/reduce.hpp>
#include <functional>
#include <vector>

std::vector<bh::algorithm::reduce_option>
get_slices(py::tuple index,
           std::function<bh::axis::index_type(bh::axis::index_type, double)> index_self,
           std::function<bh::axis::index_type(bh::axis::index_type)> size_self);

py::list expand_ellipsis(py::list indexes, py::size_t rank);
