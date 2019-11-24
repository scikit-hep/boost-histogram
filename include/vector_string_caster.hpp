// Copyright 2018-2019 Hans Dembinski and Henry Schreiner
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include "pybind11.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace pybind11 {
namespace detail {

template <>
struct type_caster<std::vector<std::string>>
    : list_caster<std::vector<std::string>, std::string> {
    using base_t = list_caster<std::vector<std::string>, std::string>;

    bool load(handle src, bool convert) {
        if(isinstance<array>(src)) {
            auto arr = reinterpret_borrow<array>(src);
            if(arr.dtype().kind() == 'S')
                return load_from_array(arr);
            if(arr.dtype().kind() == 'U' || arr.dtype().kind() == 'O') {
                try {
                    auto arr_conv = arr.attr("astype")("S");
                    return load_from_array(arr_conv);
                } catch(...) {
                }
            }
            return false;
        }

        return base_t::load(src, convert);
    }

    static std::size_t strlen(const char* p, std::size_t nmax) {
        std::size_t n = 0;
        for(; n < nmax && p[n] != 0; ++n)
            ;
        return n;
    }

    bool load_from_array(array src) {
        const auto step = static_cast<std::size_t>(src.itemsize());
        const auto size = static_cast<std::size_t>(src.size());
        auto p          = static_cast<const char*>(src.data());
        value.clear();
        value.reserve(size);
        for(std::size_t i = 0; i < size; p += step, ++i)
            value.emplace_back(std::string{p, strlen(p, step)});
        return true;
    }
};

} // namespace detail
} // namespace pybind11
