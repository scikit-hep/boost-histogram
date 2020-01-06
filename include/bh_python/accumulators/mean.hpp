// Copyright 2015-2019 Hans Dembinski and Henry Schreiner
//
// Distributed under the Boost Software License, version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Based on boost/histogram/accumulators/mean.hpp
// Changes:
//  * Internal values are public for access from Python
//  * A special constructor added for construction from Python

#pragma once

#include <boost/core/nvp.hpp>
#include <boost/histogram/weight.hpp>

namespace accumulators {

/** Calculates mean and variance of sample.

  Uses Welford's incremental algorithm to improve the numerical
  stability of mean and variance computation.
*/
template <class ValueType>
struct mean {
    using value_type      = ValueType;
    using const_reference = const value_type&;

    mean() = default;

    mean(const value_type& n,
         const value_type& mean,
         const value_type& variance) noexcept
        : count(n)
        , value(mean)
        , sum_of_deltas_squared(variance * (n - 1)) {}

    mean(const value_type& sum,
         const value_type& mean,
         const value_type& sum_of_deltas_squared,
         bool /* Tag to trigger python internal constructor */)
        : count(sum)
        , value(mean)
        , sum_of_deltas_squared(sum_of_deltas_squared) {}

    void operator()(const value_type& x) noexcept {
        count += static_cast<value_type>(1);
        const auto delta = x - value;
        value += delta / count;
        sum_of_deltas_squared += delta * (x - value);
    }

    void operator()(const boost::histogram::weight_type<value_type>& w,
                    const value_type& x) noexcept {
        count += w.value;
        const auto delta = x - value;
        value += w.value * delta / count;
        sum_of_deltas_squared += w.value * delta * (x - value);
    }

    mean& operator+=(const mean& rhs) noexcept {
        if(count != 0 || rhs.count != 0) {
            const auto tmp = value * count + rhs.value * rhs.count;
            count += rhs.count;
            value = tmp / count;
        }
        sum_of_deltas_squared += rhs.sum_of_deltas_squared;
        return *this;
    }

    mean& operator*=(const value_type& s) noexcept {
        value *= s;
        sum_of_deltas_squared *= s * s;
        return *this;
    }

    bool operator==(const mean& rhs) const noexcept {
        return count == rhs.count && value == rhs.value
               && sum_of_deltas_squared == rhs.sum_of_deltas_squared;
    }

    bool operator!=(const mean& rhs) const noexcept { return !operator==(rhs); }

    value_type variance() const noexcept { return sum_of_deltas_squared / (count - 1); }

    template <class Archive>
    void serialize(Archive& ar, unsigned) {
        ar& boost::make_nvp("count", count);
        ar& boost::make_nvp("value", value);
        ar& boost::make_nvp("sum_of_deltas_squared", sum_of_deltas_squared);
    }

    value_type count{};
    value_type value{};
    value_type sum_of_deltas_squared{};
};

} // namespace accumulators
