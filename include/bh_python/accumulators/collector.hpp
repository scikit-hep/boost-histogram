// Copyright 2020 Hans Dembinski
//
// Distributed under the Boost Software License, version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <algorithm>
#include <array>
#include <boost/core/nvp.hpp>
#include <boost/histogram/weight.hpp>
#include <vector>

namespace accumulators {

/** Keeps track of all weights in each bin.

  Can be used to compute bootstrap estimates of the uncertainies.
*/
template <class ValueType>
struct weight_collector {
    using value_type      = ValueType;
    using const_reference = const value_type&;
    using data_type       = std::vector<value_type>;

    weight_collector() = default;

    void operator+=(const boost::histogram::weight_type<value_type>& w) noexcept {
        data.push_back(w.value);
    }

    weight_collector& operator+=(const weight_collector& rhs) noexcept {
        data.reserve(data.size() + rhs.data.size());
        for(auto&& x : rhs.data)
            data.push_back(x);
        return *this;
    }

    weight_collector& operator*=(const value_type& s) noexcept {
        for(auto&& x : data)
            x *= s;
        return *this;
    }

    bool operator==(const weight_collector& rhs) const noexcept {
        return std::equal(data.begin(), data.end(), rhs.data.begin(), rhs.data.end());
    }

    bool operator!=(const weight_collector& rhs) const noexcept {
        return !operator==(rhs);
    }

    template <class Archive>
    void serialize(Archive& ar, unsigned) {
        ar& boost::make_nvp("data", data);
    }

    data_type data{};
};

/** Keeps track of all samples in each bin.

  Can be used to compute bootstrap estimates of the uncertainies.
*/
template <class ValueType>
struct sample_collector {
    using value_type      = ValueType;
    using const_reference = const value_type&;
    using item_type       = std::array<value_type, 2>;
    using data_type       = std::vector<item_type>;

    sample_collector() = default;

    void operator()(const value_type& x) noexcept { data.emplace_back(1, x); }

    void operator()(const boost::histogram::weight_type<value_type>& w,
                    const value_type& x) noexcept {
        data.emplace_back(w.value, x);
    }

    sample_collector& operator+=(const sample_collector& rhs) noexcept {
        data.reserve(data.size() + rhs.data.size());
        for(auto&& x : rhs)
            data.push_back(x);
        return *this;
    }

    sample_collector& operator*=(const value_type& s) noexcept {
        for(auto&& x : data)
            x.second *= s;
        return *this;
    }

    bool operator==(const sample_collector& rhs) const noexcept {
        return std::equal(data.begin(), data.end(), rhs.begin(), rhs.end());
    }

    bool operator!=(const sample_collector& rhs) const noexcept {
        return !operator==(rhs);
    }

    template <class Archive>
    void serialize(Archive& ar, unsigned) {
        ar& boost::make_nvp("data", data);
    }

    data_type data{};
};

} // namespace accumulators
