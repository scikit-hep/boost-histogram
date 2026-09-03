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
            auto arr        = reinterpret_borrow<array>(src);
            const auto kind = arr.dtype().kind();
            if(kind == 'S' || kind == 'U') {
                // the loaders below assume tightly packed (C-contiguous) data,
                // so make a contiguous copy if needed (e.g. sliced arrays)
                if((arr.flags() & array::c_style) == 0) {
                    arr = array::ensure(arr, array::c_style);
                    if(!arr)
                        return false;
                }
                return kind == 'S' ? load_from_array_s(arr) : load_from_array_u(arr);
            }
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

    bool load_from_array_s(const array& src) {
        const auto step = static_cast<std::size_t>(src.itemsize());
        const auto size = static_cast<std::size_t>(src.size());
        const auto* p   = static_cast<const char*>(src.data());
        value.clear();
        value.reserve(size);
        for(std::size_t i = 0; i < size; p += step, ++i)
            value.emplace_back(p, strlen(p, step));
        return true;
    }

    // encode one UCS-4 code point as UTF-8 and append it to s
    static void append_utf8(std::string& s, std::uint32_t cp) {
        if(cp < 0x80) {
            s.push_back(static_cast<char>(cp));
        } else if(cp < 0x800) {
            s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if(cp < 0x10000) {
            s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            s.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            s.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }

    bool load_from_array_u(const array& src) {
        const auto step
            = static_cast<std::size_t>(src.itemsize()) / sizeof(std::uint32_t);
        const auto size = static_cast<std::size_t>(src.size());
        const auto* p   = static_cast<const std::uint32_t*>(src.data());
        value.clear();
        value.reserve(size);
        for(std::size_t i = 0; i < size; p += step, ++i) {
            // numpy 'U' dtype stores each character as a UCS-4 code point;
            // encode to UTF-8, keeping the fast path for pure ASCII
            const auto n = strlen(p, step);
            std::string s;
            s.reserve(n);
            for(std::size_t j = 0; j < n; ++j)
                append_utf8(s, p[j]);
            value.emplace_back(std::move(s));
        }
        return true;
    }
};

} // namespace detail
} // namespace pybind11
