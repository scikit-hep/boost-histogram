// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.
//
// Based on boost/histogram/axis/ostream.hpp
// String representations here evaluate correctly in Python.

#pragma once

#include <boost/assert.hpp>
#include <boost/histogram/axis/regular.hpp>
#include <boost/histogram/detail/cat.hpp>
#include <boost/histogram/detail/static_if.hpp>
#include <boost/histogram/detail/type_name.hpp>
#include <boost/histogram/fwd.hpp>
#include <boost/throw_exception.hpp>
#include <iomanip>
#include <iosfwd>
#include <stdexcept>
#include <type_traits>

/**
  \file boost/histogram/axis/ostream.hpp
  Python streaming operators for the builtin axis types.
 **/

namespace boost {
namespace histogram {

namespace detail {
inline const char *axis_suffix(const axis::transform::id &) { return ""; }
inline const char *axis_suffix(const axis::transform::log &) { return "_log"; }
inline const char *axis_suffix(const axis::transform::sqrt &) { return "_sqrt"; }
inline const char *axis_suffix(const axis::transform::pow &) { return "_pow"; }

template <class OStream, class T>
void stream_metadata(OStream &os, const T &t) {
    detail::static_if<detail::is_streamable<T>>(
        [&os](const auto &t) {
            std::ostringstream oss;
            oss << t;
            if(!oss.str().empty()) {
                os << ", metadata=" << std::quoted(oss.str());
            }
        },
        [&os](const auto &) { os << ", metadata=" << detail::type_name<T>(); },
        t);
}

template <class OStream>
void stream_options(OStream &os, const unsigned bits) {
    bool circular  = bits & axis::option::circular;
    bool underflow = bits & axis::option::underflow;
    bool overflow  = bits & axis::option::overflow;
    bool growth    = bits & axis::option::growth;

    // Axes types (circular) that have a single flow bin should report flow=False if
    // turned off But currently, flow=False is the only supported circular axis type if
    // circular, then underflow = overflow = underflow && overflow;

    if(circular)
        return;
    else if(growth)
        os << ", growth=True";
    else if(!underflow && !overflow)
        os << ", flow=False";
    else if(underflow && !overflow)
        os << ", overflow=False";
    else if(!underflow && overflow)
        os << ", underflow=False";
}

template <class OStream, class T>
void stream_transform(OStream &, const T &) {}

template <class OStream>
void stream_transform(OStream &os, const axis::transform::pow &t) {
    os << ", power=" << t.power;
}

template <class OStream, class T>
void stream_value(OStream &os, const T &t) {
    os << t;
}

template <class OStream, class... Ts>
void stream_value(OStream &os, const std::basic_string<Ts...> &t) {
    os << std::quoted(t);
}

} // namespace detail

namespace axis {

template <class T>
class polymorphic_bin;

template <class... Ts>
std::basic_ostream<Ts...> &operator<<(std::basic_ostream<Ts...> &os,
                                      const null_type &) {
    return os; // do nothing
}

template <class... Ts, class U>
std::basic_ostream<Ts...> &operator<<(std::basic_ostream<Ts...> &os,
                                      const interval_view<U> &i) {
    os << "[" << i.lower() << ", " << i.upper() << ")";
    return os;
}

template <class... Ts, class U>
std::basic_ostream<Ts...> &operator<<(std::basic_ostream<Ts...> &os,
                                      const polymorphic_bin<U> &i) {
    if(i.is_discrete())
        os << static_cast<double>(i);
    else
        os << "[" << i.lower() << ", " << i.upper() << ")";
    return os;
}

template <class... Ts, class... Us>
std::basic_ostream<Ts...> &operator<<(std::basic_ostream<Ts...> &os,
                                      const regular<Us...> &a) {
    bool circular = a.options() & axis::option::circular;
    os << (circular ? "circular" : "regular") << detail::axis_suffix(a.transform())
       << "(" << a.size() << ", " << a.value(0) << ", " << a.value(a.size());
    detail::stream_metadata(os, a.metadata());
    detail::stream_options(os, a.options());
    detail::stream_transform(os, a.transform());
    os << ")";
    return os;
}

template <class... Ts, class... Us>
std::basic_ostream<Ts...> &operator<<(std::basic_ostream<Ts...> &os,
                                      const integer<Us...> &a) {
    os << "integer(" << a.value(0) << ", " << a.value(a.size());
    detail::stream_metadata(os, a.metadata());
    detail::stream_options(os, a.options());
    os << ")";
    return os;
}

template <class... Ts, class... Us>
std::basic_ostream<Ts...> &operator<<(std::basic_ostream<Ts...> &os,
                                      const variable<Us...> &a) {
    os << "variable([" << a.value(0);
    for(index_type i = 1, n = a.size(); i <= n; ++i) {
        os << ", " << a.value(i);
    }
    os << "]";
    detail::stream_metadata(os, a.metadata());
    detail::stream_options(os, a.options());
    os << ")";
    return os;
}

template <class... Ts, class... Us>
std::basic_ostream<Ts...> &operator<<(std::basic_ostream<Ts...> &os,
                                      const category<Us...> &a) {
    os << "category([";
    for(index_type i = 0, n = a.size(); i < n; ++i) {
        detail::stream_value(os, a.value(i));
        os << (i == (a.size() - 1) ? "" : ", ");
    }
    os << "]";
    detail::stream_metadata(os, a.metadata());
    os << ")";
    return os;
}

template <class... Ts, class... Us>
std::basic_ostream<Ts...> &operator<<(std::basic_ostream<Ts...> &os,
                                      const variant<Us...> &v) {
    visit(
        [&os](const auto &x) {
            using A = std::decay_t<decltype(x)>;
            detail::static_if<detail::is_streamable<A>>(
                [&os](const auto &x) { os << x; },
                [](const auto &) {
                    BOOST_THROW_EXCEPTION(std::runtime_error(
                        detail::cat(detail::type_name<A>(), " is not streamable")));
                },
                x);
        },
        v);
    return os;
}

} // namespace axis
} // namespace histogram
} // namespace boost
