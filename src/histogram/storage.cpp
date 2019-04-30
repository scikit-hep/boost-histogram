// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#include <boost/histogram/python/pybind11.hpp>

#include <boost/histogram.hpp>
#include <boost/histogram/python/register_storage.hpp>
#include <boost/histogram/python/storage.hpp>
#include <boost/histogram/storage_adaptor.hpp>

void register_storages(py::module &storage) {
    // Fast storages

    register_storage<storage::int_>(storage, "int", "Integers in vectors storage type");

    register_storage<storage::double_>(storage, "double", "Weighted storage without variance type (fast but simple)");

    register_storage<storage::atomic_int>(storage, "atomic_int", "Threadsafe (not growing axis) integer storage");

    // Default storages

    register_storage<storage::unlimited>(storage, "unlimited", "Optimized for unweighted histograms, adaptive");

    register_storage<storage::weight>(
        storage, "weight", "Dense storage which tracks sums of weights and a variance estimate");

    register_storage<storage::profile>(storage, "profile", "Dense storage which tracks means of samples in each cell");

    register_storage<storage::weighted_profile>(
        storage, "weighted_profile", "Dense storage which tracks means of weighted samples in each cell");
}
