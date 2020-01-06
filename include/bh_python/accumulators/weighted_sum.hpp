// Copyright 2015-2019 Hans Dembinski and Henry Schreiner
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

/// Holds sum of weights and its variance estimate
template <class ValueType>
struct weighted_sum {
    using value_type      = ValueType;
    using const_reference = const value_type&;

    weighted_sum() = default;

    /// Initialize sum to value and allow implicit conversion
    weighted_sum(const_reference value) noexcept
        : weighted_sum(value, value) {}

    /// Allow implicit conversion from sum<T>
    template <class T>
    weighted_sum(const weighted_sum<T>& s) noexcept
        : weighted_sum(s.value(), s.variance()) {}

    /// Initialize sum to value and variance
    weighted_sum(const_reference value, const_reference variance) noexcept
        : value(value)
        , variance(variance) {}

    /// Increment by one.
    weighted_sum& operator++() {
        value += 1;
        variance += 1;
        return *this;
    }

    /// Increment by weight.
    template <typename T>
    weighted_sum& operator+=(const bh::weight_type<T>& w) {
        value += w.value;
        variance += w.value * w.value;
        return *this;
    }

    /// Added another weighted sum.
    weighted_sum& operator+=(const weighted_sum& rhs) {
        value += rhs.value;
        variance += rhs.variance;
        return *this;
    }

    /// Scale by value.
    weighted_sum& operator*=(const value_type& x) {
        value *= x;
        variance *= x * x;
        return *this;
    }

    bool operator==(const value_type& rhs) const noexcept {
        return value == rhs && variance == rhs;
    }

    template <typename T>
    bool operator==(const weighted_sum<T>& rhs) const noexcept {
        return value == rhs.value && variance == rhs.variance;
    }

    template <typename T>
    bool operator!=(const T& rhs) const noexcept {
        return !operator==(rhs);
    }

    // lossy conversion must be explicit
    explicit operator const_reference() const { return value; }

    template <class Archive>
    void serialize(Archive& ar, unsigned /* version */) {
        ar& boost::make_nvp("value", value);
        ar& boost::make_nvp("variance", variance);
    }

    value_type value{};
    value_type variance{};
};

} // namespace accumulators
