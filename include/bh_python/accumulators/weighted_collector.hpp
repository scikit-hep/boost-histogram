// Copyright 2015-2019 Hans Dembinski and Henry Schreiner
//
// Distributed under the Boost Software License, version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Based on boost/histogram/accumulators/collector.hpp
//
// Changes:
//  * Collects (value, weight) pairs instead of plain values, following the
//    fill conventions of weighted_mean (sample required, weight optional)

#pragma once

#include <bh_python/accumulators/weighted_value.hpp>

#include <boost/core/nvp.hpp>
#include <boost/histogram/weight.hpp>

#include <utility>
#include <vector>

namespace accumulators {

/** Collects the original (value, weight) pairs that land in each bin.

  The weighted analog of bh::accumulators::collector: instead of aggregating,
  every fill appends the sample value and its weight to a per-bin container.
  An omitted weight is stored as 1.
*/
template <class ValueType>
struct weighted_collector {
    using value_type      = ValueType;
    using const_reference = const value_type&;

    // Each collected (value, weight) pair; see weighted_value.hpp.
    using entry = weighted_value<value_type>;

    using container_type = std::vector<entry>;
    using const_iterator = typename container_type::const_iterator;
    using size_type      = typename container_type::size_type;

    weighted_collector() = default;
    explicit weighted_collector(container_type c)
        : container_(std::move(c)) {}

    // Keep these two non-template overloads exactly in the shape of
    // weighted_mean's, so that bh::detail::accumulator_traits deduces
    // accumulator_traits_holder<true, const double&> (weighted fill, one
    // required sample).
    void operator()(const value_type& x) { operator()(boost::histogram::weight(1), x); }

    void operator()(const boost::histogram::weight_type<value_type>& w,
                    const value_type& x) {
        container_.push_back(entry{x, w.value});
    }

    /// Concatenate; drives histogram addition, project, and reduce merging.
    weighted_collector& operator+=(const weighted_collector& rhs) {
        container_.insert(container_.end(), rhs.begin(), rhs.end());
        return *this;
    }

    bool operator==(const weighted_collector& rhs) const noexcept {
        return container_ == rhs.container_;
    }
    bool operator!=(const weighted_collector& rhs) const noexcept {
        return !operator==(rhs);
    }

    size_type size() const noexcept { return container_.size(); }
    const_iterator begin() const noexcept { return container_.begin(); }
    const_iterator end() const noexcept { return container_.end(); }
    const entry& operator[](size_type i) const noexcept { return container_[i]; }
    const entry* data() const noexcept { return container_.data(); }

    template <class Archive>
    void serialize(Archive& ar, unsigned /* version */) {
        ar& boost::make_nvp("container", container_);
    }

  private:
    container_type container_;
};

} // namespace accumulators
