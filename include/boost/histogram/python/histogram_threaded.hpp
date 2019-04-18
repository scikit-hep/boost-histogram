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

#include <future>
#include <thread>

template <class Histogram>
struct [[gnu::visibility("hidden")]] fill_helper_threaded {
    fill_helper_threaded(Histogram & h, py::args args, size_t threads)
        : hist(h)
        , threads_(threads) {
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

        ssize_t ssize = arrs[0].first.size();
        for(size_t i = 1; i < dim; ++i)
            if(arrs[i].first.size() != ssize)
                throw std::invalid_argument("arrays must have same length");
        if(ssize == 0) // handle scalars by setting size to 1
            ++ssize;

        size = (size_t)ssize;
    }

    // keep function small to minimize code bloat, it is instantiated 32 times :(
    // TODO: solve this more efficiently on the lower Boost::Histogram level
    template <class N>
    void operator()(N) {
        using namespace boost::mp11;
        // N is a compile-time number with N == arrs.size()

        py::gil_scoped_release gil;

        size_t threads = threads_ == 0 ? std::thread::hardware_concurrency() : threads_;

        std::vector<std::thread> threadpool;
        std::vector<std::future<Histogram>> histpool;

        for(size_t i = 0; i < threads; i++) {
            size_t start = i * size / threads;
            size_t stop  = (i + 1 == threads) ? size : (i + 1) * size / threads;
            std::packaged_task<Histogram()> task([start, stop, this]() {
                Histogram local_hist(hist);
                local_hist.reset();

                // Type: tuple<double, ..., double> (N times)
                mp_repeat<std::tuple<double>, N> tp;

                for(std::size_t i = start; i < stop; ++i) {
                    mp_for_each<mp_iota<N>>([&](auto I) {
                        // I is mp_size_t<0>, mp_size_t<1>, ..., mp_size_t<N-1>
                        std::get<I>(tp) = arrs[I].second[i];
                    });
                    local_hist(tp); // throws invalid_argument if tup has wrong size
                }
                return local_hist;
            });

            histpool.push_back(task.get_future());
            threadpool.emplace_back(std::move(task));
        }

        for(auto &hp : histpool)
            hist += hp.get();

        for(auto &t : threadpool)
            t.join();
    }

    // specialization for N=0 to prevent compile-time error in histogram
    void operator()(boost::mp11::mp_size_t<0>) { throw std::invalid_argument("at least one argument required"); }

    // specialization for N=1, implement potential optimizations here
    void operator()(boost::mp11::mp_size_t<1>) {
        // compilers often emit faster code for range-based for loops
        struct span {
            const double *begin() const { return start; }
            const double *end() const { return stop; }
            const double *start;
            const double *stop;
        };

        // Note that splitting and filling from multiple threads is only supported with atomics
        py::gil_scoped_release gil;

        size_t threads = threads_ == 0 ? std::thread::hardware_concurrency() : threads_;

        std::vector<std::thread> threadpool;
        std::vector<std::future<Histogram>> histpool;

        for(size_t i = 0; i < threads; i++) {
            size_t start = i * size / threads;
            size_t stop  = (i + 1 == threads) ? size : (i + 1) * size / threads;
            std::packaged_task<Histogram()> task([start, stop, this]() {
                Histogram local_hist(hist);
                local_hist.reset();

                for(double xi : span{arrs.front().second + start, arrs.front().second + stop})
                    local_hist(xi); // throws invalid_argument if tup has wrong size

                return local_hist;
            });

            histpool.push_back(task.get_future());
            threadpool.emplace_back(std::move(task));
        }

        for(auto &hp : histpool)
            hist += hp.get();

        for(auto &t : threadpool)
            t.join();
    }

    Histogram &hist;
    std::vector<std::pair<py::array_t<double>, const double *>> arrs;
    size_t size;
    size_t threads_;
};
