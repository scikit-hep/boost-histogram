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
template <typename RealType>
struct weighted_sum {
    weighted_sum() = default;
    explicit weighted_sum(const RealType& value) noexcept
        : value(value)
        , variance(value) {}
    weighted_sum(const RealType& value, const RealType& variance) noexcept
        : value(value)
        , variance(variance) {}

    /// Increment by one.
    weighted_sum& operator++() { return operator+=(1); }

    /// Increment by value.
    template <typename T>
    weighted_sum& operator+=(const T& val) {
        value += val;
        variance += val * val;
        return *this;
    }

    /// Added another weighted sum.
    template <typename T>
    weighted_sum& operator+=(const weighted_sum<T>& rhs) {
        value += static_cast<RealType>(rhs.value);
        variance += static_cast<RealType>(rhs.variance);
        return *this;
    }

    /// Scale by value.
    weighted_sum& operator*=(const RealType& x) {
        value *= x;
        variance *= x * x;
        return *this;
    }

    bool operator==(const RealType& rhs) const noexcept {
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
    template <class T>
    explicit operator T() const {
        return static_cast<T>(value);
    }

    template <class Archive>
    void serialize(Archive& ar, unsigned /* version */) {
        ar& boost::make_nvp("value", value);
        ar& boost::make_nvp("variance", variance);
    }

    RealType value    = RealType();
    RealType variance = RealType();
};

} // namespace accumulators
