// Copyright 2018-2019 Hans Dembinski and Henry Schreiner
//
// Distributed under the Boost Software License, version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Based on boost/histogram/accumulators/weighted_mean.hpp
//
// Changes:
//  * Internal values are public for access from Python
//  * A special constructor added for construction from Python

#pragma once

#include <boost/core/nvp.hpp>
#include <boost/histogram/weight.hpp>

namespace accumulators {

/**
  Calculates mean and variance of weighted sample.

  Uses West's incremental algorithm to improve numerical stability
  of mean and variance computation.
*/
template <typename ValueType>
struct weighted_mean {
    using value_type      = ValueType;
    using const_reference = const value_type&;

    weighted_mean() = default;

    weighted_mean(const value_type& wsum,
                  const value_type& wsum2,
                  const value_type& mean,
                  const value_type& variance)
        : sum_of_weights(wsum)
        , sum_of_weights_squared(wsum2)
        , value(mean)
        , _sum_of_weighted_deltas_squared(
              variance * (sum_of_weights - sum_of_weights_squared / sum_of_weights)) {}

    weighted_mean(const value_type& wsum,
                  const value_type& wsum2,
                  const value_type& mean,
                  const value_type& _sum_of_weighted_deltas_squared,
                  bool /* tag to trigger Python internal constructor */)
        : sum_of_weights(wsum)
        , sum_of_weights_squared(wsum2)
        , value(mean)
        , _sum_of_weighted_deltas_squared(_sum_of_weighted_deltas_squared) {}

    void operator()(const value_type& x) { operator()(boost::histogram::weight(1), x); }

    void operator()(const boost::histogram::weight_type<value_type>& w,
                    const value_type& x) {
        sum_of_weights += w.value;
        sum_of_weights_squared += w.value * w.value;
        const auto delta = x - value;
        value += w.value * delta / sum_of_weights;
        _sum_of_weighted_deltas_squared += w.value * delta * (x - value);
    }

    weighted_mean& operator+=(const weighted_mean& rhs) {
        if(sum_of_weights != 0 || rhs.sum_of_weights != 0) {
            const auto tmp = value * sum_of_weights + rhs.value * rhs.sum_of_weights;
            sum_of_weights += rhs.sum_of_weights;
            sum_of_weights_squared += rhs.sum_of_weights_squared;
            value = tmp / sum_of_weights;
        }
        _sum_of_weighted_deltas_squared += rhs._sum_of_weighted_deltas_squared;
        return *this;
    }

    weighted_mean& operator*=(const value_type& s) {
        value *= s;
        _sum_of_weighted_deltas_squared *= s * s;
        return *this;
    }

    bool operator==(const weighted_mean& rhs) const noexcept {
        return sum_of_weights == rhs.sum_of_weights
               && sum_of_weights_squared == rhs.sum_of_weights_squared
               && value == rhs.value
               && _sum_of_weighted_deltas_squared
                      == rhs._sum_of_weighted_deltas_squared;
    }

    bool operator!=(const weighted_mean rhs) const noexcept { return !operator==(rhs); }

    value_type variance() const {
        return _sum_of_weighted_deltas_squared
               / (sum_of_weights - sum_of_weights_squared / sum_of_weights);
    }

    template <class Archive>
    void serialize(Archive& ar, unsigned /* version */) {
        ar& boost::make_nvp("sum_of_weights", sum_of_weights);
        ar& boost::make_nvp("sum_of_weights_squared", sum_of_weights_squared);
        ar& boost::make_nvp("value", value);
        ar& boost::make_nvp("_sum_of_weighted_deltas_squared",
                            _sum_of_weighted_deltas_squared);
    }

    value_type sum_of_weights{};
    value_type sum_of_weights_squared{};
    value_type value{};
    value_type _sum_of_weighted_deltas_squared{};
};

} // namespace accumulators
