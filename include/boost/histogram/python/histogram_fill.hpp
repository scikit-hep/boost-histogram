// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <boost/histogram/python/pybind11.hpp>

#include <boost/histogram.hpp>

#include <boost/mp11.hpp>

#include <cassert>
#include <cmath>
#include <tuple>
#include <vector>

template <class Histogram>
struct [[gnu::visibility("hidden")]] fill_helper {
    fill_helper(Histogram & h, py::args args)
        : hist(h) {
        size_t dim = args.size();
        if(dim == 0)
            throw std::invalid_argument("at least one argument required");

        arrs.reserve(dim);
        for(size_t i = 0; i < dim; ++i) {
            auto a = py::cast<py::array_t<double>>(args[i]);
            if(a.ndim() > 1)
                throw std::invalid_argument("array dim must be 0 or 1");
            arrs.emplace_back(a, a.data());
        }

        size = arrs[0].first.size();
        for(size_t i = 1; i < dim; ++i)
            if(arrs[i].first.size() != size)
                throw std::invalid_argument("arrays must have same length");
        if(size == 0) // handle scalars by setting size to 1
            ++size;
    }

    // keep function small to minimize code bloat, it is instantiated 32 times :(
    // TODO: solve this more efficiently on the lower Boost::Histogram level
    template <class N>
    void operator()(N) {
        using namespace boost::mp11;
        // N is a compile-time number with N == arrs.size()
        // Type: tuple<double, ..., double> (N times)
        mp_repeat<std::tuple<double>, N> tp;

        // Note that splitting and filling from multiple threads is only supported with atomics
        py::gil_scoped_release gil;

        for(std::size_t i = 0; i < (std::size_t)size; ++i) {
            mp_for_each<mp_iota<N>>([&](auto I) {
                // I is mp_size_t<0>, mp_size_t<1>, ..., mp_size_t<N-1>
                std::get<I>(tp) = arrs[I].second[i];
            });
            hist(tp); // throws invalid_argument if tup has wrong size
        }
    }

    // specialization for N=0 to prevent compile-time error in histogram
    void operator()(boost::mp11::mp_size_t<0>) { throw std::invalid_argument("at least one argument required"); }

    // specialization for N=1, implement potential optimizations here
    void operator()(boost::mp11::mp_size_t<1>) {
        // compilers often emit faster code for range-based for loops
        struct span {
            const double *begin() const { return ptr; }
            const double *end() const { return ptr + size; }
            const double *ptr;
            std::size_t size;
        };

        // Note that splitting and filling from multiple threads is only supported with atomics
        py::gil_scoped_release gil;

        for(double xi : span{arrs.front().second, (std::size_t)size})
            hist(xi); // throws invalid_argument if hist.rank() != 1
    }

    Histogram &hist;
    std::vector<std::pair<py::array_t<double>, const double *>> arrs;
    ssize_t size;
};
