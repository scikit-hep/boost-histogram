// Copyright 2015-2019 Henry Schreiner and Hans Dembinski
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

template <class CharT, class Traits, class T>
std::basic_ostream<CharT, Traits>&
handle_nonzero_width(std::basic_ostream<CharT, Traits>& os, const T& x) {
    const auto w = os.width();
    os.width(0);
    boost::histogram::detail::counting_streambuf<CharT, Traits> cb;
    const auto saved = os.rdbuf(&cb);
    os << x;
    os.rdbuf(saved);
    if(os.flags() & std::ios::left) {
        os << x;
        for(auto i = cb.count; i < w; ++i)
            os << os.fill();
    } else {
        for(auto i = cb.count; i < w; ++i)
            os << os.fill();
        os << x;
    }
    return os;
}

// Note that the names are *not* included here, so they can be added in Pybind11.

// namespace boost {
// namespace histogram {
// namespace accumulators {
// template <class CharT, class Traits, class W>
// std::basic_ostream<CharT, Traits>&
// operator<<(std::basic_ostream<CharT, Traits>& os,
//           const sum<W>& x) {
//    if(os.width() == 0)
//        return os << x.large() << " + " << x.small();
//    return handle_nonzero_width(os, x);
//}
//
//} // namespace accumulators
//} // namespace histogram
//} // namespace boost

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
