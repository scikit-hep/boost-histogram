// Copyright 2015-2019 Hans Dembinski and Henry Schreiner
//
// Distributed under the Boost Software License, version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <boost/core/nvp.hpp>

#include <type_traits>

namespace accumulators {

/** A single collected (value, weight) pair, the element type of weighted_collector.

  It is a plain aggregate: standard-layout and trivially copyable, so a contiguous
  ``std::vector<weighted_value>`` copies directly into a packed
  ``[('value', 'f8'), ('weight', 'f8')]`` structured numpy array, which is what the
  weighted collector's ``view()`` exposes per bin.
*/
template <class ValueType>
struct weighted_value {
    using value_type = ValueType;

    value_type value{};
    value_type weight{};

    weighted_value() = default;
    weighted_value(value_type value_, value_type weight_) noexcept
        : value(value_)
        , weight(weight_) {}

    bool operator==(const weighted_value& rhs) const noexcept {
        return value == rhs.value && weight == rhs.weight;
    }
    bool operator!=(const weighted_value& rhs) const noexcept {
        return !operator==(rhs);
    }

    template <class Archive>
    void serialize(Archive& ar, unsigned /* version */) {
        ar& boost::make_nvp("value", value);
        ar& boost::make_nvp("weight", weight);
    }
};

static_assert(std::is_standard_layout<weighted_value<double>>::value
                  && std::is_trivially_copyable<weighted_value<double>>::value
                  && sizeof(weighted_value<double>) == 2 * sizeof(double),
              "weighted_value must be two packed doubles");

} // namespace accumulators
