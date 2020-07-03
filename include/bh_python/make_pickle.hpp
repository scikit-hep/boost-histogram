// Copyright 2015-2019 Hans Dembinski and Henry Schreiner
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Design notes: This uses the Boost.Serialization mechanism without actually using the
// Boost.Serialization library. The Boost mechanism is an algorithm to recursively break
// down complex objects into primitives for saving and doing the reverse for loading.
// When we save, we are building a tuple of Python primitives with this mechanism and
// when we load, we are doing the reserve. Large buffers of primitive types are best
// stored by converting them to numpy arrays. This way, we can ride piggy-back on
// numpy's pickle code for fast loading and saving.

// The Boost approach stores very little meta-data, only a version number. It relies on
// the fact that the internal structure of the C++ classes is not changed. When you
// actually change this internal structure, you must also increment this version
// number and add code to the affected serialize method to load the previous version and
// the new version.
//
// It is very important that there are specializations for all storages with non-trivial
// accumulators. A histogram can have very very many cells and the generic serializer
// converts each cell individually into Python objects which is extremely time
// consuming. The storages should be specialized in the corresponding header file that
// includes the storages, by providing a specialization of the save and load functions
//
// template <class Archive>
// void save(Archive& ar, const MyClass& c, unsigned version);
//
// template <class Archive>
// void load(Archive& ar, const MyClass& c, unsigned version);
//
// or for classes with template arguments, something like
//
// template <class Archive, class T0, class T1>
// void save(Archive& ar, const MyClass<T0, T1>& c, unsigned version);
//
// template <class Archive, class T0, class T1>
// void load(Archive& ar, MyClass<T0, T1>& c, unsigned version);
//
// in the global namespace. It should not be necessary to touch the code here.

#pragma once

#include <bh_python/pybind11.hpp>

#include <bh_python/metadata.hpp>

#include <boost/assert.hpp>
#include <boost/core/nvp.hpp>
#include <boost/histogram/detail/array_wrapper.hpp>
#include <boost/mp11/function.hpp> // mp_or
#include <boost/mp11/utility.hpp>  // mp_valid

#include <algorithm>
#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

template <class T,
          class = decltype(
              std::declval<T&>().serialize(std::declval<std::nullptr_t&>(), 0))>
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

template <class T>
struct is_string : std::false_type {};

template <class C>
struct is_string<std::basic_string<C>> : std::true_type {};

template <class C, class T>
struct is_string<std::basic_string<C, T>> : std::true_type {};

template <class T>
using is_serialization_primitive =
    typename boost::mp11::mp_or<std::is_arithmetic<T>, is_string<T>>::type;

template <class Archive, class T>
void save(Archive& ar, const T& t, unsigned version) {
    // default implementation calls serialize method
    const_cast<T&>(t).serialize(ar, version);
}

template <class Archive, class T>
void load(Archive& ar, T& t, unsigned version) {
    // default implementation calls serialize method
    static_assert(std::is_const<T>::value == false, "T must be non-const");
    t.serialize(ar, version);
}

template <class Archive, class T>
void split_serialize(std::true_type, Archive& ar, T& t, unsigned version) {
    load(ar, t, version);
}

template <class Archive, class T>
void split_serialize(std::false_type, Archive& ar, const T& t, unsigned version) {
    save(ar, t, version);
}

template <class Archive, class T>
void serialize(Archive& ar, T& t, unsigned version) {
    // default implementation calls serialize method
    static_assert(std::is_const<T>::value == false, "T must be non-const");
    split_serialize(typename Archive::is_loading{}, ar, t, version);
}

// builds a tuple of Python primitives from C++ primitives
class tuple_oarchive {
  public:
    using is_saving  = std::true_type;
    using is_loading = std::false_type;

    tuple_oarchive(py::tuple& tup)
        : tup_(tup) {}

    template <class T>
    tuple_oarchive& operator&(boost::nvp<T> t) {
        return operator<<(t.const_value());
    }

    template <class T>
    tuple_oarchive& operator<<(boost::nvp<T> t) {
        return operator<<(t.const_value());
    }

    template <class T>
    tuple_oarchive& operator&(const T& t) {
        return operator<<(t);
    }

    template <class T>
    std::enable_if_t<is_serialization_primitive<T>::value == true, tuple_oarchive&>
    operator<<(const T& t) {
        // no version number is saved for primitives
        this->operator<<(py::cast(t));
        return *this;
    }

    template <class T>
    std::enable_if_t<is_serialization_primitive<T>::value == false, tuple_oarchive&>
    operator<<(const T& t) {
        // save current class version with each non-primitive type
        const unsigned version = boost::serialization::version<T>::value;
        this->operator<<(version);
        serialize(*this, const_cast<T&>(t), version);
        return *this;
    }

    tuple_oarchive& operator<<(const py::object& obj) {
        // maybe use growth factor 1.6 and shrink tuple to final size in destructor?
        tup_ = tup_ + py::make_tuple(obj);
        return *this;
    }

    tuple_oarchive& operator<<(py::object& obj) {
        return operator<<(static_cast<const py::object&>(obj));
    }

    tuple_oarchive& operator<<(py::object&& obj) {
        return operator<<(static_cast<const py::object&>(obj));
    }

    // put specializations here that side-step normal serialization
    // tuple_oarchive& operator<<(py::str& m) {
    //     return operator<<(static_cast<py::object&>(m));
    // }

    tuple_oarchive& operator<<(const py::str& m) {
        return operator<<(static_cast<const py::object&>(m));
    }

    tuple_oarchive& operator<<(const metadata_t& m) {
        return operator<<(static_cast<const py::object&>(m));
    }

    template <class T>
    tuple_oarchive& operator<<(const py::array_t<T>& a) {
        return operator<<(static_cast<const py::object&>(a));
    }

    template <class T>
    std::enable_if_t<std::is_arithmetic<T>::value == true, tuple_oarchive&>
    operator<<(const std::vector<T>& v) {
        // fast version for vector of arithmetic types
        py::array_t<T> a(v.size(), v.data());
        this->operator<<(static_cast<const py::object&>(a));
        return *this;
    }

    template <class T>
    std::enable_if_t<std::is_arithmetic<T>::value == false, tuple_oarchive&>
    operator<<(const std::vector<T>& v) {
        // generic version
        this->operator<<(v.size());
        for(auto&& item : v)
            this->operator<<(item);
        return *this;
    }

    template <class T>
    std::enable_if_t<std::is_arithmetic<T>::value == true, tuple_oarchive&>
    operator<<(const bh::detail::array_wrapper<T>& w) {
        // fast version
        py::array_t<T> a(w.size, w.ptr);
        this->operator<<(static_cast<const py::object&>(a));
        return *this;
    }

    template <class T>
    std::enable_if_t<std::is_arithmetic<T>::value == false, tuple_oarchive&>
    operator<<(const bh::detail::array_wrapper<T>& w) {
        // generic version
        for(auto&& item : bh::detail::make_span(w.ptr, w.size))
            this->operator<<(item);
        return *this;
    }

  private:
    py::tuple& tup_;
};

class tuple_iarchive {
  public:
    using is_saving  = std::false_type;
    using is_loading = std::true_type;

    tuple_iarchive(const py::tuple& t)
        : tup_(t) {}

    // no object tracking
    void reset_object_address(const void*, const void*){};

    template <class T>
    tuple_iarchive& operator&(boost::nvp<T> t) {
        return operator>>(t.value());
    }

    template <class T>
    tuple_iarchive& operator>>(boost::nvp<T> t) {
        return operator>>(t.value());
    }

    template <class T>
    tuple_iarchive& operator&(T& t) {
        return operator>>(t);
    }

    template <class T>
    std::enable_if_t<is_serialization_primitive<T>::value == true, tuple_iarchive&>
    operator>>(T& t) {
        // no version number is saved for primitives
        py::object obj;
        this->operator>>(obj);
        t = py::cast<T>(obj);
        return *this;
    }

    template <class T>
    std::enable_if_t<is_serialization_primitive<T>::value == false, tuple_iarchive&>
    operator>>(T& t) {
        // load saved class version of each non-primitive type to call legacy code
        unsigned saved_version;
        this->operator>>(saved_version);
        serialize(*this, t, saved_version);
        return *this;
    }

    tuple_iarchive& operator>>(py::object& obj) {
        BOOST_ASSERT(cur_ < tup_.size());
        obj = tup_[cur_++];
        return *this;
    }

    // put specializations here that side-step normal serialization

    tuple_iarchive& operator>>(py::str& m) {
        return operator>>(static_cast<py::object&>(m));
    }

    tuple_iarchive& operator>>(metadata_t& m) {
        return operator>>(static_cast<py::object&>(m));
    }

    template <class T>
    tuple_iarchive& operator>>(py::array_t<T>& a) {
        return operator>>(static_cast<py::object&>(a));
    }

    template <class T>
    std::enable_if_t<std::is_arithmetic<T>::value == true, tuple_iarchive&>
    operator>>(std::vector<T>& v) {
        // fast version for vector of arithmetic types
        py::array_t<T> a;
        this->operator>>(a);
        v.resize(static_cast<std::size_t>(a.size()));
        // sadly we cannot move the memory from the numpy array into the vector
        std::copy(a.data(), a.data() + a.size(), v.begin());
        return *this;
    }

    template <class T>
    std::enable_if_t<std::is_arithmetic<T>::value == false, tuple_iarchive&>
    operator>>(std::vector<T>& v) {
        // generic version
        std::size_t new_size;
        this->operator>>(new_size);
        v.resize(new_size);
        for(auto&& item : v)
            this->operator>>(item);
        return *this;
    }

    template <class T>
    std::enable_if_t<std::is_arithmetic<T>::value == true, tuple_iarchive&>
    operator>>(bh::detail::array_wrapper<T>& w) {
        // fast version
        py::array_t<T> a;
        this->operator>>(a);
        // buffer wrapped by array_wrapper must already have correct size
        BOOST_ASSERT(static_cast<std::size_t>(a.size()) == w.size);
        // sadly we cannot move the memory from the numpy array into the vector
        std::copy(a.data(), a.data() + a.size(), w.ptr);
        return *this;
    }

    template <class T>
    std::enable_if_t<std::is_arithmetic<T>::value == false, tuple_iarchive&>
    operator>>(bh::detail::array_wrapper<T>& w) {
        // generic version
        for(auto&& item : bh::detail::make_span(w.ptr, w.size))
            this->operator>>(item);
        return *this;
    }

  private:
    const py::tuple& tup_;
    std::size_t cur_ = 0;
};

/// Make a pickle serializer with a given type
template <class T>
decltype(auto) make_pickle() {
    return py::pickle(
        [](const T& obj) {
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
