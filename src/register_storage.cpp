// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#include <bh_python/pybind11.hpp>

#include <bh_python/register_storage.hpp>
#include <bh_python/storage.hpp>
#include <boost/histogram/storage_adaptor.hpp>

void register_storages(py::module& storage) {
    register_storage<storage::int64>(
        storage, "int64", "Integers in vectors storage type");

    register_storage<storage::double_>(
        storage, "double", "Weighted storage without variance type (fast but simple)");

    register_storage<storage::atomic_int64>(
        storage, "atomic_int64", "Threadsafe (not growing axis) integer storage");

    register_storage<storage::unlimited>(
        storage, "unlimited", "Optimized for unweighted histograms, adaptive");

    register_storage<storage::weight>(
        storage,
        "weight",
        "Dense storage which tracks sums of weights and a variance estimate");

    register_storage<storage::mean>(
        storage, "mean", "Dense storage which tracks means of samples in each cell");

    register_storage<storage::weighted_mean>(
        storage,
        "weighted_mean",
        "Dense storage which tracks means of weighted samples in each cell");
}
