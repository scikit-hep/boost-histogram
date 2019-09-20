// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <boost/histogram/accumulators/ostream.hpp>
#include <boost/histogram/fwd.hpp>
#include <boost/histogram/python/axis_ostream.hpp>
#include <boost/histogram/python/storage.hpp>
#include <iosfwd>

namespace boost {
namespace histogram {

template <typename CharT, typename Traits, typename A, typename S>
std::basic_ostream<CharT, Traits> &operator<<(std::basic_ostream<CharT, Traits> &os,
                                              const histogram<A, S> &h) {
    os << "histogram(";
    h.for_each_axis([&](const auto &a) { os << "\n  " << a << ","; });
    os << (h.rank() ? "\n  " : " ") << "storage=" << storage::name<S>();
    os << (h.rank() ? "\n)" : ")");
    return os;
}

} // namespace histogram
} // namespace boost
