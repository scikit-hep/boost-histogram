#pragma once

namespace boost {
namespace histogram {
namespace python {

/// This is pretty much identical to the C++20 definition
/// or the boost::type_traits helper, but available for C++14+.
template <class T>
struct remove_cvref {
    using type = std::remove_cv_t<std::remove_reference_t<T>>;
};

/// Type version of remove_cvref
template <class T>
using remove_cvref_t = typename remove_cvref<T>::type;

} // namespace python
} // namespace histogram
} // namespace boost
