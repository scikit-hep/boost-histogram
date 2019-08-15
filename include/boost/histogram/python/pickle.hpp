// Copyright 2015-2019 Hans Dembinski and Henry Schreiner
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <boost/histogram/detail/detect.hpp>
#include <boost/histogram/python/pybind11.hpp>

#include <boost/assert.hpp>
#include <boost/mp11/tuple.hpp>

#include <type_traits>

struct OutToTuple;
struct InFromTuple;

namespace boost {
namespace histogram {
BOOST_HISTOGRAM_DETECT(has_method_serialize, (std::declval<T &>().serialize(std::declval<OutToTuple &>(), 0)));
BOOST_HISTOGRAM_DETECT(has_function_serialize, (serialize(std::declval<OutToTuple &>(), std::declval<T &>(), 0)));
} // namespace histogram
} // namespace boost

struct OutToTuple {
    using is_loading = std::false_type;
    py::tuple tuple;

    template <class T,
              std::enable_if_t<bh::has_method_serialize<T>::value && !bh::has_function_serialize<T>::value> * = nullptr>
    OutToTuple &operator&(T &&arg) {
        arg.serialize(*this, 0);
        return *this;
    }

    template <class T,
              std::enable_if_t<!bh::has_method_serialize<T>::value && bh::has_function_serialize<T>::value> * = nullptr>
    OutToTuple &operator&(T &&arg) {
        serialize(*this, arg, 0);
        return *this;
    }

    template <
        typename T,
        std::enable_if_t<!bh::has_method_serialize<T>::value && !bh::has_function_serialize<T>::value> * = nullptr>
    OutToTuple &operator&(T &&arg) {
        tuple = tuple + py::make_tuple<py::return_value_policy::reference>(arg);
        return *this;
    }
};

struct InFromTuple {
    using is_loading = std::true_type;
    const py::tuple &tuple;
    size_t current = 0;

    InFromTuple(const py::tuple &tuple_)
        : tuple(tuple_) {}

    template <class T,
              std::enable_if_t<bh::has_method_serialize<T>::value && !bh::has_function_serialize<T>::value> * = nullptr>
    InFromTuple &operator&(T &&arg) {
        arg.serialize(*this, 0);
        return *this;
    }

    template <class T,
              std::enable_if_t<!bh::has_method_serialize<T>::value && bh::has_function_serialize<T>::value> * = nullptr>
    InFromTuple &operator&(T &&arg) {
        serialize(*this, arg, 0);
        return *this;
    }

    template <
        typename T,
        std::enable_if_t<!bh::has_method_serialize<T>::value && !bh::has_function_serialize<T>::value> * = nullptr>
    InFromTuple &operator&(T &&arg) {
        using Tbase = std::decay_t<T>;
        arg         = py::cast<Tbase>(tuple[current++]);
        return *this;
    }
};

/// Make a pickle serializer with a given type
template <class T>
decltype(auto) make_pickle() {
    return py::pickle(
        [](const T &p) {
            OutToTuple out;
            out &const_cast<T &>(p);
            return out.tuple;
        },
        [](py::tuple t) {
            InFromTuple in{t};
            T p;
            in &p;
            return p;
        });
}

// This allows the serialization header to be as close as possible to the official one
namespace serialization {

template <class T>
decltype(auto) make_nvp(const char *, T &&item) {
    return item;
}

} // namespace serialization
