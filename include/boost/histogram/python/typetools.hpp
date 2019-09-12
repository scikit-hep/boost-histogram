// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

namespace boost {
namespace histogram {
namespace python {

/// Identical to the C++20 definition
template <class T>
using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

} // namespace python
} // namespace histogram
} // namespace boost
