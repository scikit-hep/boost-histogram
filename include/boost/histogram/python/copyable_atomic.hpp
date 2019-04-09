// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <atomic>

template <typename T>
class copyable_atomic : public std::atomic<T> {
public:
    using std::atomic<T>::atomic;
    
    // zero-initialize the atomic T
    copyable_atomic() noexcept : std::atomic<T>(T()) {}
    
    // this is potentially not thread-safe, see below
    copyable_atomic(const copyable_atomic& rhs) : std::atomic<T>() { this->operator=(rhs); }
    
    // this is potentially not thread-safe, see below
    copyable_atomic& operator=(const copyable_atomic& rhs) {
        if (this != &rhs) { std::atomic<T>::store(rhs.load()); }
        return *this;
    }
    
    // Default to relaxed memory order
    T operator++() noexcept {
        return this->fetch_add(T(1), std::memory_order_relaxed); // Should return +1 but result unused
    }
};
