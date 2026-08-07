// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <bh_python/pybind11.hpp>

// Each function lives in its own translation unit (src/register_hist_*.cpp)
// so the heavy per-storage template instantiations compile in parallel and
// keep per-process memory low.

void register_histogram_int64(py::module& hist);
void register_histogram_unlimited(py::module& hist);
void register_histogram_double(py::module& hist);
void register_histogram_atomic_int64(py::module& hist);
void register_histogram_weight(py::module& hist);
void register_histogram_mean(py::module& hist);
void register_histogram_weighted_mean(py::module& hist);
void register_histogram_multi_cell(py::module& hist);
