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

#include <boost/assert.hpp>
#include <boost/core/nvp.hpp>
#include <boost/histogram/fwd.hpp> // for mean<>
#include <boost/throw_exception.hpp>
#include <stdexcept>
#include <type_traits>

namespace boost {
namespace histogram {
namespace python {

/** Calculates mean and variance of sample.

  Uses Welfords's incremental algorithm to improve the numerical
  stability of mean and variance computation.
*/
template <class RealType>
struct mean {
  public:
    mean() = default;
    mean(const RealType &n, const RealType &mean, const RealType &variance) noexcept
        : sum_(n)
        , mean_(mean)
        , sum_of_deltas_squared_(variance * (n - 1)) {}

    mean(const RealType &sum,
         const RealType &mean,
         const RealType &sum_of_deltas_squared,
         bool /* Tag to trigger python internal constructor */)
        : sum_(sum)
        , mean_(mean)
        , sum_of_deltas_squared_(sum_of_deltas_squared) {}

    void operator()(const RealType &x) noexcept {
        sum_ += static_cast<RealType>(1);
        const auto delta = x - mean_;
        mean_ += delta / sum_;
        sum_of_deltas_squared_ += delta * (x - mean_);
    }

    void operator()(const weight_type<RealType> &w, const RealType &x) noexcept {
        sum_ += w.value;
        const auto delta = x - mean_;
        mean_ += w.value * delta / sum_;
        sum_of_deltas_squared_ += w.value * delta * (x - mean_);
    }

    template <class T>
    mean &operator+=(const mean<T> &rhs) noexcept {
        if(sum_ != 0 || rhs.sum_ != 0) {
            const auto tmp = mean_ * sum_ + static_cast<RealType>(rhs.mean_ * rhs.sum_);
            sum_ += rhs.sum_;
            mean_ = tmp / sum_;
        }
        sum_of_deltas_squared_ += static_cast<RealType>(rhs.sum_of_deltas_squared_);
        return *this;
    }

    mean &operator*=(const RealType &s) noexcept {
        mean_ *= s;
        sum_of_deltas_squared_ *= s * s;
        return *this;
    }

    template <class T>
    bool operator==(const mean<T> &rhs) const noexcept {
        return sum_ == rhs.sum_ && mean_ == rhs.mean_
               && sum_of_deltas_squared_ == rhs.sum_of_deltas_squared_;
    }

    template <class T>
    bool operator!=(const mean<T> &rhs) const noexcept {
        return !operator==(rhs);
    }

    const RealType &count() const noexcept { return sum_; }
    const RealType &value() const noexcept { return mean_; }
    RealType variance() const noexcept { return sum_of_deltas_squared_ / (sum_ - 1); }

    template <class Archive>
    void serialize(Archive &ar, unsigned version) {
        if(version == 0) {
            // read only
            std::size_t sum;
            ar &make_nvp("sum", sum);
            sum_ = static_cast<RealType>(sum);
        } else {
            ar &make_nvp("sum", sum_);
        }
        ar &make_nvp("mean", mean_);
        ar &make_nvp("sum_of_deltas_squared", sum_of_deltas_squared_);
    }

    RealType sum_ = 0, mean_ = 0, sum_of_deltas_squared_ = 0;
};

} // namespace python
} // namespace histogram
} // namespace boost

#ifndef BOOST_HISTOGRAM_DOXYGEN_INVOKED

namespace boost {
namespace serialization {

template <class T>
struct version;

// version 1 for boost::histogram::accumulators::mean<RealType>
template <class RealType>
struct version<histogram::python::mean<RealType>> : std::integral_constant<int, 1> {};

} // namespace serialization
} // namespace boost

namespace std {
template <class T, class U>
/// Specialization for boost::histogram::accumulators::mean.
struct common_type<boost::histogram::accumulators::mean<T>,
                   boost::histogram::accumulators::mean<U>> {
    using type = boost::histogram::accumulators::mean<common_type_t<T, U>>;
};
} // namespace std

#endif
