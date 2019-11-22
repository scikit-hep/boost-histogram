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
template <typename RealType>
struct weighted_mean {
    weighted_mean() = default;
    weighted_mean(const RealType& wsum,
                  const RealType& wsum2,
                  const RealType& mean,
                  const RealType& variance)
        : sum_of_weights(wsum)
        , sum_of_weights_squared(wsum2)
        , value(mean)
        , sum_of_weighted_deltas_squared(
              variance * (sum_of_weights - sum_of_weights_squared / sum_of_weights)) {}

    weighted_mean(const RealType& wsum,
                  const RealType& wsum2,
                  const RealType& mean,
                  const RealType& sum_of_weighted_deltas_squared,
                  bool)
        : sum_of_weights(wsum)
        , sum_of_weights_squared(wsum2)
        , value(mean)
        , sum_of_weighted_deltas_squared(sum_of_weighted_deltas_squared) {}

    void operator()(const RealType& x) { operator()(boost::histogram::weight(1), x); }

    void operator()(const boost::histogram::weight_type<RealType>& w,
                    const RealType& x) {
        sum_of_weights += w.value;
        sum_of_weights_squared += w.value * w.value;
        const auto delta = x - value;
        value += w.value * delta / sum_of_weights;
        sum_of_weighted_deltas_squared += w.value * delta * (x - value);
    }

    template <typename T>
    weighted_mean& operator+=(const weighted_mean<T>& rhs) {
        if(sum_of_weights != 0 || rhs.sum_of_weights != 0) {
            const auto tmp = value * sum_of_weights
                             + static_cast<RealType>(rhs.value * rhs.sum_of_weights);
            sum_of_weights += static_cast<RealType>(rhs.sum_of_weights);
            sum_of_weights_squared += static_cast<RealType>(rhs.sum_of_weights_squared);
            value = tmp / sum_of_weights;
        }
        sum_of_weighted_deltas_squared
            += static_cast<RealType>(rhs.sum_of_weighted_deltas_squared);
        return *this;
    }

    weighted_mean& operator*=(const RealType& s) {
        value *= s;
        sum_of_weighted_deltas_squared *= s * s;
        return *this;
    }

    template <typename T>
    bool operator==(const weighted_mean<T>& rhs) const noexcept {
        return sum_of_weights == rhs.sum_of_weights
               && sum_of_weights_squared == rhs.sum_of_weights_squared
               && value == rhs.value
               && sum_of_weighted_deltas_squared == rhs.sum_of_weighted_deltas_squared;
    }

    template <typename T>
    bool operator!=(const T& rhs) const noexcept {
        return !operator==(rhs);
    }

    RealType variance() const {
        return sum_of_weighted_deltas_squared
               / (sum_of_weights - sum_of_weights_squared / sum_of_weights);
    }

    template <class Archive>
    void serialize(Archive& ar, unsigned /* version */) {
        ar& boost::make_nvp("sum_of_weights", sum_of_weights);
        ar& boost::make_nvp("sum_of_weights_squared", sum_of_weights_squared);
        ar& boost::make_nvp("value", value);
        ar& boost::make_nvp("sum_of_weighted_deltas_squared",
                            sum_of_weighted_deltas_squared);
    }

    RealType sum_of_weights = RealType(), sum_of_weights_squared = RealType(),
             value = RealType(), sum_of_weighted_deltas_squared = RealType();
};

} // namespace accumulators
