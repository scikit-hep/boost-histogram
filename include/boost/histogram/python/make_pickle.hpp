// Copyright 2015-2019 Hans Dembinski and Henry Schreiner
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <boost/histogram/python/pybind11.hpp>

#include <boost/assert.hpp>
#include <boost/core/nvp.hpp>
#include <boost/mp11/function.hpp> // mp_or
#include <boost/mp11/utility.hpp>  // mp_valid
#include <type_traits>
#include <utility>
#include <vector>

template <class T,
          class = decltype(std::declval<T>().serialize(std::declval<nullptr_t &>(), 0))>
struct has_method_serialize_impl {};

template <class T>
using has_method_serialize =
    typename boost::mp11::mp_valid<has_method_serialize_impl, T>::type;

namespace boost {
namespace serialization {
// provide default implementation of boost::serialization::version
template <class>
struct version : std::integral_constant<int, 0> {};
} // namespace serialization
} // namespace boost

template <class Archive, class T>
void serialize(Archive &ar, std::vector<T> &t, unsigned /* version */) {
    auto size = t.size();
    ar &size;
    if(Archive::is_loading::value)
        t.resize(size);
    for(auto &&ti : t)
        ar &ti;
}

template <class T>
using is_serialization_primitive =
    typename boost::mp11::mp_or<std::is_arithmetic<T>,
                                std::is_same<T, std::string>,
                                std::is_same<T, char *>>::type;

// saving
template <class Archive, class T>
void serialize_primitive(Archive &ar, T &t, std::false_type) {
    auto obj = py::cast(t);
    ar << obj;
}

// loading
template <class Archive, class T>
void serialize_primitive(Archive &ar, T &t, std::true_type) {
    py::object obj;
    ar >> obj;
    t = py::cast<T>(obj);
}

template <class Archive, class T>
void serialize_impl(Archive &ar, T &t, unsigned, std::false_type) {
    static_assert(is_serialization_primitive<T>::value, "");
    serialize_primitive(ar, t, typename Archive::is_loading{});
}

template <class Archive, class T>
void serialize_impl(Archive &ar, T &t, unsigned version, std::true_type) {
    t.serialize(ar, version);
}

template <class Archive, class T>
void serialize(Archive &ar, T &t, unsigned version) {
    static_assert(std::is_const<T>::value == false, "");
    serialize_impl(ar, t, version, has_method_serialize<T>{});
}

// builds a tuple of Python primitives from C++ primitives
struct tuple_oarchive {
    using is_saving  = std::true_type;
    using is_loading = std::false_type;

    py::tuple &tup;

    template <class T>
    tuple_oarchive &operator&(boost::nvp<T> t) {
        return operator<<(t.const_value());
    }

    template <class T>
    tuple_oarchive &operator<<(boost::nvp<T> t) {
        return operator<<(t.const_value());
    }

    template <class T>
    tuple_oarchive &operator&(const T &t) {
        return operator<<(t);
    }

    template <class T>
    tuple_oarchive &operator<<(const T &t) {
        serialize(*this, const_cast<T &>(t), boost::serialization::version<T>::value);
        return *this;
    }

    tuple_oarchive &operator<<(py::object obj) {
        // TODO: use growth factor 1.6 and shrink tuple to final size in destructor
        tup = tup + py::make_tuple(obj);
        return *this;
    }
};

struct tuple_iarchive {
    using is_saving  = std::false_type;
    using is_loading = std::true_type;

    const py::tuple &tup_;
    std::size_t cur_ = 0;

    tuple_iarchive(const py::tuple &t)
        : tup_(t) {}

    template <class T>
    tuple_iarchive &operator&(boost::nvp<T> t) {
        return operator>>(t.value());
    }

    template <class T>
    tuple_iarchive &operator>>(boost::nvp<T> t) {
        return operator>>(t.value());
    }

    template <class T>
    tuple_iarchive &operator&(T &t) {
        return operator>>(t);
    }

    template <class T>
    tuple_iarchive &operator>>(T &t) {
        serialize(*this, t, boost::serialization::version<T>::value);
        return *this;
    }

    tuple_iarchive &operator>>(py::object &obj) {
        BOOST_ASSERT(cur_ < tup_.size());
        obj = tup_[cur_++];
        return *this;
    }

    // no object tracking
    void reset_object_address(const void *, const void *){};
};

/// Make a pickle serializer with a given type
template <class T>
decltype(auto) make_pickle() {
    return py::pickle(
        [](const T &obj) {
            py::tuple tup;
            tuple_oarchive oa{tup};
            oa << obj;
            return tup;
        },
        [](py::tuple tup) {
            tuple_iarchive ia{tup};
            T obj;
            ia >> obj;
            return obj;
        });
}
