// Copyright 2018-2019 Hans Dembinski and Henry Schreiner
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <bh_python/pybind11.hpp>

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
                return load_from_array_s(arr);
            if(arr.dtype().kind() == 'U')
                return load_from_array_u(arr);
        }
        return base_t::load(src, convert);
    }

    template <class T>
    static std::size_t strlen(const T* p, std::size_t nmax) {
        std::size_t n = 0;
        for(; n < nmax && p[n] != 0; ++n)
            ;
        return n;
    }

    bool load_from_array_s(array src) {
        const auto step = static_cast<std::size_t>(src.itemsize());
        const auto size = static_cast<std::size_t>(src.size());
        auto p          = static_cast<const char*>(src.data());
        value.clear();
        value.reserve(size);
        for(std::size_t i = 0; i < size; p += step, ++i)
            value.emplace_back(std::string{p, strlen(p, step)});
        return true;
    }

    bool load_from_array_u(array src) {
        const auto step
            = static_cast<std::size_t>(src.itemsize()) / sizeof(std::uint32_t);
        const auto size = static_cast<std::size_t>(src.size());
        auto p          = static_cast<const std::uint32_t*>(src.data());
        value.clear();
        value.reserve(size);
        for(std::size_t i = 0; i < size; p += step, ++i) {
            // check that UTF-32 only contains ASCII, fail if not
            const auto n = strlen(p, step);
            std::string s;
            s.reserve(n);
            for(std::size_t i = 0; i < n; ++i) {
                if(p[i] >= 128)
                    return false;
                s.push_back(static_cast<char>(p[i]));
            }
            value.emplace_back(s);
        }
        return true;
    }
};

} // namespace detail
} // namespace pybind11
