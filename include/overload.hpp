// Copyright 2018-2019 Hans Dembinski and Henry Schreiner
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <utility> // for std::forward

template <class...>
struct overload_t;

template <class F>
struct overload_t<F> : F {
    overload_t(F&& f)
        : F(std::forward<F>(f)) {}
    using F::operator();
};

template <class F, class... Fs>
struct overload_t<F, Fs...> : F, overload_t<Fs...> {
    overload_t(F&& x, Fs&&... xs)
        : F(std::forward<F>(x))
        , overload_t<Fs...>(std::forward<Fs>(xs)...) {}
    using F::operator();
    using overload_t<Fs...>::operator();
};

template <class... Fs>
auto overload(Fs&&... xs) {
    return overload_t<Fs...>(std::forward<Fs>(xs)...);
}
