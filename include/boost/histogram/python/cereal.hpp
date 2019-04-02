// Copyright 2015-2019 Hans Dembinski and Henry Schreiner
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_CEREAL_HPP
#define BOOST_HISTOGRAM_CEREAL_HPP

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

#include <cereal/cereal.hpp>

#include <cereal/types/array.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/boost_variant.hpp>
#include <cereal/types/vector.hpp>

// Non-portable across endianess
#include <cereal/archives/binary.hpp>

//#include <boost/serialization/array.hpp>
//#include <boost/serialization/map.hpp>
//#include <boost/serialization/nvp.hpp>
//#include <boost/serialization/string.hpp>
//#include <boost/serialization/variant.hpp>
//#include <boost/serialization/vector.hpp>

#include <tuple>
#include <type_traits>

/// Pickle a Boost::Histogram serializable object and 0+ python objects
template<typename T, typename... Args>
py::tuple pickle_totuple(const T &p, Args&&... args){
    std::stringstream data;
    cereal::BinaryOutputArchive archive( data );
    archive(p);
    return py::make_tuple(py::bytes(data.str()), std::forward<Args>(args)...);
}

/// Convert the first element in a tuple to a Boost::Histogram object
template<typename T>
T pickle_fromtuple(py::tuple t){
    std::stringstream data;
    data << py::cast<std::string>(t[0]);
    cereal::BinaryInputArchive archive( data );
    T p;
    archive(p);
    return p;
}


namespace boost {
namespace histogram {

namespace accumulators {
template <class RealType>
template <class Archive>
void sum<RealType>::serialize(Archive& ar, unsigned /* version */) {
  ar& cereal::make_nvp("large", large_);
  ar& cereal::make_nvp("small", small_);
}

template <class RealType>
template <class Archive>
void weighted_sum<RealType>::serialize(Archive& ar, unsigned /* version */) {
  ar& cereal::make_nvp("sum_of_weights", sum_of_weights_);
  ar& cereal::make_nvp("sum_of_weights_squared", sum_of_weights_squared_);
}

template <class RealType>
template <class Archive>
void mean<RealType>::serialize(Archive& ar, unsigned /* version */) {
  ar& cereal::make_nvp("sum", sum_);
  ar& cereal::make_nvp("mean", mean_);
  ar& cereal::make_nvp("sum_of_deltas_squared", sum_of_deltas_squared_);
}

template <class RealType>
template <class Archive>
void weighted_mean<RealType>::serialize(Archive& ar, unsigned /* version */) {
  ar& cereal::make_nvp("sum_of_weights", sum_of_weights_);
  ar& cereal::make_nvp("sum_of_weights_squared", sum_of_weights_squared_);
  ar& cereal::make_nvp("weighted_mean", weighted_mean_);
  ar& cereal::make_nvp("sum_of_weighted_deltas_squared",
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
  ar& cereal::make_nvp("power", t.power);
}
} // namespace transform

template <class Archive>
void serialize(Archive&, null_type&, unsigned /* version */) {}

template <class T, class Tr, class M, class O>
template <class Archive>
void regular<T, Tr, M, O>::serialize(Archive& ar, unsigned /* version */) {
  ar& cereal::make_nvp("transform", static_cast<transform_type&>(*this));
  ar& cereal::make_nvp("size", size_meta_.first());
  // ar& cereal::make_nvp("meta", size_meta_.second());  // Python stores metadata seperately
  ar& cereal::make_nvp("min", min_);
  ar& cereal::make_nvp("delta", delta_);
}

template <class T, class M, class O>
template <class Archive>
void integer<T, M, O>::serialize(Archive& ar, unsigned /* version */) {
  ar& cereal::make_nvp("size", size_meta_.first());
  //ar& cereal::make_nvp("meta", size_meta_.second());
  ar& cereal::make_nvp("min", min_);
}

template <class T, class M, class O, class A>
template <class Archive>
void variable<T, M, O, A>::serialize(Archive& ar, unsigned /* version */) {
  ar& cereal::make_nvp("seq", vec_meta_.first());
  //ar& cereal::make_nvp("meta", vec_meta_.second());
}

template <class T, class M, class O, class A>
template <class Archive>
void category<T, M, O, A>::serialize(Archive& ar, unsigned /* version */) {
  ar& cereal::make_nvp("seq", vec_meta_.first());
  //ar& cereal::make_nvp("meta", vec_meta_.second());
}

template <class... Ts>
template <class Archive>
void variant<Ts...>::serialize(Archive& ar, unsigned /* version */) {
  ar& cereal::make_nvp("variant", impl);
}
} // namespace axis

namespace detail {
template <class Archive, class T>
void serialize(Archive& ar, vector_impl<T>& impl, unsigned /* version */) {
  ar& cereal::make_nvp("vector", static_cast<T&>(impl));
}

template <class Archive, class T>
void serialize(Archive& ar, array_impl<T>& impl, unsigned /* version */) {
  ar& cereal::make_nvp("size", impl.size_);
  ar& cereal::make_nvp("array", impl);
                       // cereal::binary_data(&impl.front(), impl.size_));
}

template <class Archive, class T>
void serialize(Archive& ar, map_impl<T>& impl, unsigned /* version */) {
  ar& cereal::make_nvp("size", impl.size_);
  ar& cereal::make_nvp("map", static_cast<T&>(impl));
}

template <class Archive, class Allocator>
void serialize(Archive& ar, mp_int<Allocator>& x, unsigned /* version */) {
  ar& cereal::make_nvp("data", x.data);
}
} // namespace detail

template <class Archive, class T>
void serialize(Archive& ar, storage_adaptor<T>& s, unsigned /* version */) {
  ar& cereal::make_nvp("impl", static_cast<detail::storage_adaptor_impl<T>&>(s));
}
    
template <class A>
template <class Archive>
void unlimited_storage<A>::serialize(Archive& ar, unsigned /* version */) {
  if (Archive::is_loading::value) {
    buffer_type dummy(buffer.alloc);
    std::size_t size;
    ar& cereal::make_nvp("type", dummy.type);
    ar& cereal::make_nvp("size", size);
    dummy.apply([this, size](auto* tp) {
      BOOST_ASSERT(tp == nullptr);
      using T = detail::remove_cvref_t<decltype(*tp)>;
      buffer.template make<T>(size);
    });
  } else {
    ar& cereal::make_nvp("type", buffer.type);
    ar& cereal::make_nvp("size", buffer.size);
  }
  buffer.apply([this, &ar](auto*) {
    // using T = detail::remove_cvref_t<decltype(*tp)>;
    ar& cereal::make_nvp("buffer", buffer);
        //cereal::binary_data(reinterpret_cast<T*>(buffer.ptr), buffer.size));
  });
}

template <class Archive, class A, class S>
void serialize(Archive& ar, histogram<A, S>& h, unsigned /* version */) {
  ar& cereal::make_nvp("axes", unsafe_access::axes(h));
  ar& cereal::make_nvp("storage", unsafe_access::storage(h));
}

} // namespace histogram
} // namespace boost

#endif
