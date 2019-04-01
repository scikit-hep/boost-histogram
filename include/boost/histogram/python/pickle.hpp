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


struct OutToTuple {
    py::tuple tuple;
    
    template<typename... Args>
    void operator()(Args&&... args) {
        tuple = tuple + py::make_tuple(std::forward<Args>(args)...);
    }
};

struct InFromTuple {
    const py::tuple& tuple;
    size_t current = 0;
    
    InFromTuple(const py::tuple& tuple_) : tuple(tuple_) {}
    
    template<typename T, typename... Args>
    void operator()(T&& arg, Args&&... args) {
        using Tbase = std::decay_t<T>;
        arg = py::cast<Tbase>(tuple[current++]);
        operator()(std::forward<Args>(args)...);
    }
    
    void operator()() {}
};

/// Convert "serialize" to something usable for Python
template<typename T>
py::tuple to_py_tuple(const T& p) {
    OutToTuple out;
    const_cast<T&>(p).serialize(out,0);
    return out.tuple;
}

/// Convert "serialize" to something usable for Python
template<typename T>
T from_py_tuple(const py::tuple& t) {
    InFromTuple in{t};
    T p;
    p.serialize(in, 0);
    return p;
}

/// Make a pickle serializer with a given type
template<typename T>
decltype(auto) make_pickle() {
    return py::pickle(
        [](const T &p){ return to_py_tuple(p); },
        [](py::tuple t){return from_py_tuple<T>(t); });
}

// Note that this is just designed this way to make accessing private members easy.
// Python does all the serialization.

namespace boost {
namespace histogram {

namespace accumulators {

template <class RealType>
template <class Archive>
void sum<RealType>::serialize(Archive& ar, unsigned /* version */) {
    ar(large_, small_);
}

template <class RealType>
template <class Archive>
void weighted_sum<RealType>::serialize(Archive& ar, unsigned /* version */) {
    ar(sum_of_weights_, sum_of_weights_squared_);
}

template <class RealType>
template <class Archive>
void mean<RealType>::serialize(Archive& ar, unsigned /* version */) {
    ar(sum_, mean_, sum_of_deltas_squared_);
}

template <class RealType>
template <class Archive>
void weighted_mean<RealType>::serialize(Archive& ar, unsigned /* version */) {
    ar(sum_of_weights_,
                        sum_of_weights_squared_,
                        weighted_mean_,
                        sum_of_weighted_deltas_squared_);
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
  ar(t.power);
}
} // namespace transform

template <class Archive>
void serialize(Archive&, null_type&, unsigned /* version */) {}

template <class T, class Tr, class M, class O>
template <class Archive>
void regular<T, Tr, M, O>::serialize(Archive& ar, unsigned /* version */) {
  transform::serialize(ar, static_cast<transform_type&>(*this), 0);
  ar(size_meta_.first(), size_meta_.second(), min_, delta_);
}

template <class T, class M, class O>
template <class Archive>
void integer<T, M, O>::serialize(Archive& ar, unsigned /* version */) {
  ar(size_meta_.first());
  ar(size_meta_.second());
  ar(min_);
}

template <class T, class M, class O, class A>
template <class Archive>
void variable<T, M, O, A>::serialize(Archive& ar, unsigned /* version */) {
  ar(vec_meta_.first());
  ar(vec_meta_.second());
}

template <class T, class M, class O, class A>
template <class Archive>
void category<T, M, O, A>::serialize(Archive& ar, unsigned /* version */) {
  ar(vec_meta_.first());
  ar(vec_meta_.second());
}

template <class... Ts>
template <class Archive>
void variant<Ts...>::serialize(Archive& ar, unsigned /* version */) {
  ar(impl);
}
} // namespace axis
//
//namespace detail {
//template <class Archive, class T>
//void serialize(Archive& ar, vector_impl<T>& impl, unsigned /* version */) {
//  ar& cereal::make_nvp("vector", static_cast<T&>(impl));
//}
//
//template <class Archive, class T>
//void serialize(Archive& ar, array_impl<T>& impl, unsigned /* version */) {
//  ar& cereal::make_nvp("size", impl.size_);
//  ar& cereal::make_nvp("array", impl);
//                       // cereal::binary_data(&impl.front(), impl.size_));
//}
//
//template <class Archive, class T>
//void serialize(Archive& ar, map_impl<T>& impl, unsigned /* version */) {
//  ar& cereal::make_nvp("size", impl.size_);
//  ar& cereal::make_nvp("map", static_cast<T&>(impl));
//}
//
//template <class Archive, class Allocator>
//void serialize(Archive& ar, mp_int<Allocator>& x, unsigned /* version */) {
//  ar& cereal::make_nvp("data", x.data);
//}
//} // namespace detail
//
//template <class Archive, class T>
//void serialize(Archive& ar, storage_adaptor<T>& s, unsigned /* version */) {
//  ar& cereal::make_nvp("impl", static_cast<detail::storage_adaptor_impl<T>&>(s));
//}
//
//template <class A>
//template <class Archive>
//void unlimited_storage<A>::serialize(Archive& ar, unsigned /* version */) {
//  if (Archive::is_loading::value) {
//    buffer_type dummy(buffer.alloc);
//    std::size_t size;
//    ar& cereal::make_nvp("type", dummy.type);
//    ar& cereal::make_nvp("size", size);
//    dummy.apply([this, size](auto* tp) {
//      BOOST_ASSERT(tp == nullptr);
//      using T = detail::remove_cvref_t<decltype(*tp)>;
//      buffer.template make<T>(size);
//    });
//  } else {
//    ar& cereal::make_nvp("type", buffer.type);
//    ar& cereal::make_nvp("size", buffer.size);
//  }
//  buffer.apply([this, &ar](auto* tp) {
//    using T = detail::remove_cvref_t<decltype(*tp)>;
//    ar& cereal::make_nvp("buffer", buffer);
//        //cereal::binary_data(reinterpret_cast<T*>(buffer.ptr), buffer.size));
//  });
//}
//
//template <class Archive, class A, class S>
//void serialize(Archive& ar, histogram<A, S>& h, unsigned /* version */) {
//  ar& cereal::make_nvp("axes", unsafe_access::axes(h));
//  ar& cereal::make_nvp("storage", unsafe_access::storage(h));
//}

} // namespace histogram
} // namespace boost

#endif
