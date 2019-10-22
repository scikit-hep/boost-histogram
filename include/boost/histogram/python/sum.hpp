// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <boost/histogram/algorithm/sum.hpp>
#include <boost/mp11.hpp>
#include <type_traits>

template <class histogram_t>
decltype(auto) sum_histogram(const histogram_t &self, bool flow) {
    if(flow) {
        return bh::algorithm::sum(self);
    } else {
        using T       = typename histogram_t::value_type;
        using AddType = boost::mp11::mp_if<std::is_arithmetic<T>, double, T>;
        using Sum     = boost::mp11::
            mp_if<std::is_arithmetic<T>, bh::accumulators::sum<double>, T>;
        Sum sum;
        for(auto &&x : bh::indexed(self))
            sum += (AddType)*x;
        using R = boost::mp11::mp_if<std::is_arithmetic<T>, double, T>;
        return static_cast<R>(sum);
    }
}
