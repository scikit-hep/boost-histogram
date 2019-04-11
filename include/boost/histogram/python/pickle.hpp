// Copyright 2015-2019 Hans Dembinski and Henry Schreiner
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_PICKLE_HPP
#define BOOST_HISTOGRAM_PICKLE_HPP

#include <boost/assert.hpp>
#include <boost/histogram/accumulators/mean.hpp>
#include <boost/histogram/accumulators/sum.hpp>
#include <boost/histogram/accumulators/weighted_mean.hpp>
#include <boost/histogram/accumulators/weighted_sum.hpp>
#include <boost/histogram/axis/category.hpp>
#include <boost/histogram/axis/integer.hpp>
#include <boost/histogram/axis/regular.hpp>
#include <boost/histogram/axis/variable.hpp>
#include <boost/histogram/axis/variant.hpp>
#include <boost/histogram/histogram.hpp>
#include <boost/histogram/storage_adaptor.hpp>
#include <boost/histogram/unlimited_storage.hpp>
#include <boost/histogram/unsafe_access.hpp>
#include <boost/mp11/tuple.hpp>

#include <boost/histogram/python/pybind11.hpp>

#include <type_traits>

struct OutToTuple;
struct InFromTuple;

namespace boost { namespace histogram {
    BOOST_HISTOGRAM_DETECT(has_method_serialize, (std::declval<T&>().serialize(std::declval<OutToTuple&>(), 0)));
    BOOST_HISTOGRAM_DETECT(has_function_serialize, (serialize(std::declval<OutToTuple&>(), std::declval<T&>(), 0)));
}}

struct OutToTuple {
    using is_loading = std::false_type;
    py::tuple tuple;
    
    template<typename T, std::enable_if_t<bh::has_method_serialize<T>::value && !bh::has_function_serialize<T>::value>* = nullptr>
    OutToTuple& operator& (T&& arg) {
        arg.serialize(*this, 0);
        return *this;
    }
    
    template<typename T, std::enable_if_t<!bh::has_method_serialize<T>::value && bh::has_function_serialize<T>::value>* = nullptr>
    OutToTuple& operator& (T&& arg) {
        serialize(*this, arg, 0);
        return *this;
    }
    
    
    template<typename T, std::enable_if_t<!bh::has_method_serialize<T>::value && !bh::has_function_serialize<T>::value>* = nullptr>
    OutToTuple& operator& (T&& arg) {
        tuple = tuple + py::make_tuple<py::return_value_policy::reference>(arg);
        return *this;
    }
};

struct InFromTuple {
    using is_loading = std::true_type;
    const py::tuple& tuple;
    size_t current = 0;
    
    InFromTuple(const py::tuple& tuple_) : tuple(tuple_) {}
    
    template<typename T, std::enable_if_t<bh::has_method_serialize<T>::value && !bh::has_function_serialize<T>::value>* = nullptr>
    InFromTuple& operator& (T&& arg) {
        arg.serialize(*this, 0);
        return *this;
    }
    
    template<typename T, std::enable_if_t<!bh::has_method_serialize<T>::value && bh::has_function_serialize<T>::value>* = nullptr>
    InFromTuple& operator& (T&& arg) {
        serialize(*this, arg, 0);
        return *this;
    }
    
    template<typename T, std::enable_if_t<!bh::has_method_serialize<T>::value && !bh::has_function_serialize<T>::value>* = nullptr>
    InFromTuple& operator& (T&& arg) {
        using Tbase = std::decay_t<T>;
        arg = py::cast<Tbase>(tuple[current++]);
        return *this;
    }
};


/// Make a pickle serializer with a given type
template<typename T>
decltype(auto) make_pickle() {
    return py::pickle(
        [](const T &p){
            OutToTuple out;
            out & const_cast<T&>(p);
            return out.tuple;
        },
        [](py::tuple t){
            InFromTuple in{t};
            T p;
            in & p;
            return p;
        });
}

// Note that this is just designed this way to make accessing private members easy.
// Python does all the serialization.

namespace boost {
namespace histogram {

namespace accumulators {

template <class RealType>
template <class Archive>
void sum<RealType>::serialize(Archive& ar, unsigned /* version */) {
    ar & large_ & small_;
}

template <class RealType>
template <class Archive>
void weighted_sum<RealType>::serialize(Archive& ar, unsigned /* version */) {
    ar & sum_of_weights_ & sum_of_weights_squared_;
}

template <class RealType>
template <class Archive>
void mean<RealType>::serialize(Archive& ar, unsigned /* version */) {
    ar & sum_ & mean_ & sum_of_deltas_squared_;
}

template <class RealType>
template <class Archive>
void weighted_mean<RealType>::serialize(Archive& ar, unsigned /* version */) {
    ar & sum_of_weights_
       & sum_of_weights_squared_
       & weighted_mean_
       & sum_of_weighted_deltas_squared_;
}
} // namespace accumulators

namespace axis {

namespace transform {
template <class Archive>
void serialize(Archive&, id&, unsigned /* version */) {}

template <class Archive>
void serialize(Archive&, log&, unsigned /* version */) {}

template <class Archive>
void serialize(Archive&, sqrt&, unsigned /* version */) {}

template <class Archive>
void serialize(Archive& ar, pow& t, unsigned /* version */) {
  ar & t.power;
}
} // namespace transform

template <class Archive>
void serialize(Archive&, null_type&, unsigned /* version */) {}

template <class T, class Tr, class M, class O>
template <class Archive>
void regular<T, Tr, M, O>::serialize(Archive& ar, unsigned /* version */) {
    ar & static_cast<transform_type&>(*this);
    ar & size_meta_.first() & size_meta_.second() & min_ & delta_;
}

template <class T, class M, class O>
template <class Archive>
void integer<T, M, O>::serialize(Archive& ar, unsigned /* version */) {
    ar & size_meta_.first() & size_meta_.second() & min_;
}

template <class T, class M, class O, class A>
template <class Archive>
void variable<T, M, O, A>::serialize(Archive& ar, unsigned /* version */) {
    ar & vec_meta_.first() & vec_meta_.second();
}

template <class T, class M, class O, class A>
template <class Archive>
void category<T, M, O, A>::serialize(Archive& ar, unsigned /* version */) {
    ar & vec_meta_.first() & vec_meta_.second();
}

template <class... Ts>
template <class Archive>
void variant<Ts...>::serialize(Archive& ar, unsigned /* version */) {
    ar & impl;
}
} // namespace axis

namespace detail {
template <class Archive, class T>
void serialize(Archive& ar, vector_impl<T>& impl, unsigned /* version */) {
    ar & static_cast<T&>(impl);
}

template <class Archive, class T>
void serialize(Archive& ar, array_impl<T>& impl, unsigned /* version */) {
    ar & impl.size_ & impl;
}

template <class Archive, class T>
void serialize(Archive& ar, map_impl<T>& impl, unsigned /* version */) {
    ar & impl.size_ & static_cast<T&>(impl);
}

template <class Archive, class Allocator>
void serialize(Archive& ar, mp_int<Allocator>& x, unsigned /* version */) {
    ar & x.data;
}
} // namespace detail

template <class Archive, class T>
void serialize(Archive& ar, storage_adaptor<T>& s, unsigned /* version */) {
    ar & static_cast<detail::storage_adaptor_impl<T>&>(s);
}

template <class A>
template <class Archive>
void unlimited_storage<A>::serialize(Archive& ar, unsigned /* version */) {
  if (Archive::is_loading::value) {
    buffer_type dummy(buffer.alloc);
    std::size_t size;
      
    ar & dummy.type & size;
      
    dummy.apply([this, size](auto* tp) {
      BOOST_ASSERT(tp == nullptr);
      using T = detail::remove_cvref_t<decltype(*tp)>;
      buffer.template make<T>(size);
    });
  } else {
    ar & buffer.type & buffer.size;
  }
    
  buffer.apply([this, &ar](auto* tp) {
    ar & buffer;
  });
}

template <class Archive, class A, class S>
void serialize(Archive& ar, histogram<A, S>& h, unsigned /* version */) {
  ar & unsafe_access::axes(h) & unsafe_access::storage(h);
}

} // namespace histogram
} // namespace boost

#endif
