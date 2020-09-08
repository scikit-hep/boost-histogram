// Copyright 2015-2020 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.
//
// Based on boost/histogram/axis/ostream.hpp
// String representations here evaluate correctly in Python.

#pragma once

#include <bh_python/accumulators/mean.hpp>
#include <bh_python/accumulators/weighted_mean.hpp>
#include <bh_python/accumulators/weighted_sum.hpp>

#include <boost/histogram/accumulators/sum.hpp>
#include <boost/histogram/detail/counting_streambuf.hpp>
#include <boost/histogram/fwd.hpp>
#include <iosfwd>

/**
  \file boost/histogram/accumulators/ostream.hpp
  Simple streaming operators for the builtin accumulator types.
  Mostly similer to boost/histogram/accumulators/ostream.hpp
 */

template <class CharT, class Traits, class T>
std::basic_ostream<CharT, Traits>&
handle_nonzero_width(std::basic_ostream<CharT, Traits>& os, const T& x) {
    const auto w = os.width();
    os.width(0);
    std::streamsize count = 0;
    {
        auto g = ::boost::histogram::detail::make_count_guard(os, count);
        os << x;
    }
    if(os.flags() & std::ios::left) {
        os << x;
        for(auto i = count; i < w; ++i)
            os << os.fill();
    } else {
        for(auto i = count; i < w; ++i)
            os << os.fill();
        os << x;
    }
    return os;
}

// Note that the names are *not* included here, so they can be added in pybind11.

namespace accumulators {

template <class CharT, class Traits, class W>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os,
                                              const weighted_sum<W>& x) {
    if(os.width() == 0)
        return os << "value=" << x.value << ", variance=" << x.variance;
    return handle_nonzero_width(os, x);
}

template <class CharT, class Traits, class W>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os,
                                              const mean<W>& x) {
    if(os.width() == 0)
        return os << "count=" << x.count << ", value=" << x.value
                  << ", variance=" << x.variance();
    return handle_nonzero_width(os, x);
}

template <class CharT, class Traits, class W>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os,
                                              const weighted_mean<W>& x) {
    if(os.width() == 0)
        return os << "sum_of_weights=" << x.sum_of_weights
                  << ", sum_of_weights_squared=" << x.sum_of_weights_squared
                  << ", value=" << x.value << ", variance=" << x.variance();
    return handle_nonzero_width(os, x);
}

template <class CharT, class Traits, class T>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& os,
           const ::boost::histogram::accumulators::thread_safe<T>& x) {
    os << x.load();
    return os;
}

} // namespace accumulators
