// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <bh_python/metadata.hpp>

#include <boost/core/nvp.hpp>
#include <boost/histogram/axis/regular.hpp>

namespace bh = boost::histogram;

namespace axis {

/// Mimics the numpy behavoir exactly.
class regular_numpy : public bh::axis::regular<double, bh::use_default, metadata_t> {
    using value_type = double;
    double stop_{0};

  public:
    regular_numpy(unsigned n, value_type start, value_type stop, metadata_t meta = {})
        : regular(n, start, stop, meta)
        , stop_(stop) {}

    regular_numpy()
        : regular() {}

    bh::axis::index_type index(value_type v) const {
        return v <= stop_ ? std::min(regular::index(v), size() - 1) : regular::index(v);
    }

    template <class Archive>
    void serialize(Archive& ar, unsigned version) {
        regular::serialize(ar, version);
        ar& boost::make_nvp("stop", stop_);
    }
};

} // namespace axis
