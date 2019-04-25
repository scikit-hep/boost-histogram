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

#include <thread>

template <class Histogram>
struct [[gnu::visibility("hidden")]] fill_helper {
    Histogram *hist;
    std::shared_ptr<Histogram> local_hist;        // Only used when threading - empty otherwise
    std::vector<py::array_t<double>> source_ptrs; // Stores a copy of converted values *if* conversion is neccesary
    std::vector<const double *> data_ptrs;        // The actually data is stored here
    ssize_t size = 1;                             // How many values to copy
    size_t dim;                                   // Current number of dimensions to fill

    fill_helper(Histogram & h, py::args args)
        : hist(&h) {
        dim = args.size();
        if(dim == 0)
            throw std::invalid_argument("at least one argument required");

        source_ptrs.reserve(dim);
        data_ptrs.reserve(dim);

        for(size_t i = 0; i < dim; ++i) {
            auto a = py::cast<py::array_t<double>>(args[i]);

            if(a.ndim() > 1)
                throw std::invalid_argument("array dim must be 0 or 1");

            if(size != 1 && size != a.size())
                throw std::invalid_argument("arrays must have same length");
            size = a.size();

            // We keep the python objects just in case a conversion was made
            // and the Python reference counting is needed.
            source_ptrs.emplace_back(a);

            data_ptrs.emplace_back(a.data());
        }
    }

    /// Make an atomic copy (same histogram internally)
    fill_helper<Histogram> make_atomic(ssize_t total_threads, ssize_t current_thread) {
        fill_helper<Histogram> self{*this};

        ssize_t start = current_thread * size / total_threads;
        ssize_t stop  = (current_thread + 1 == total_threads) ? size : (current_thread + 1) * size / total_threads;
        self.size     = stop - start; // Start is always less than or equal to stop

        for(auto &item : self.data_ptrs) {
            item += start;
        }

        return self;
    }

    /// Make a threaded copy (new histogram internally)
    fill_helper<Histogram> make_threaded(ssize_t total_threads, ssize_t current_thread) {
        fill_helper<Histogram> self = make_atomic(total_threads, current_thread);

        // Make a new "optional" local histogram, and make the hist pointer point at it instead of the master
        self.local_hist.reset(new Histogram(*hist));
        self.hist = self.local_hist.get();

        return self;
    }

    // keep function small to minimize code bloat, it is instantiated 16 times :(
    // TODO: solve this more efficiently on the lower Boost::Histogram level
    template <class N>
    void operator()(N) {
        // N is a compile-time number with N == arrs.size()
        // Type: tuple<double, ..., double> (N times)
        boost::mp11::mp_repeat<std::tuple<double>, N> tp;
        using m_list = boost::mp11::mp_iota<N>; // list of integral constants

        for(std::size_t i = 0; i < (std::size_t)size; ++i) {
            boost::mp11::mp_for_each<m_list>([&](auto I) {
                // I is mp_size_t<0>, mp_size_t<1>, ..., mp_size_t<N-1>
                std::get<I>(tp) = data_ptrs[I][i];
            });
            (*hist)(tp); // throws invalid_argument if tup has wrong size
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

        for(double xi : span{data_ptrs.front(), (std::size_t)size})
            (*hist)(xi); // throws invalid_argument if hist.rank() != 1
    }

    // Standard fill
    void fill() {
        py::gil_scoped_release tmp;
        boost::mp11::mp_with_index<BOOST_HISTOGRAM_DETAIL_AXES_LIMIT>(dim, std::ref(*this));
    }

    // Threaded fill
    void fill_threaded(ssize_t * threads_ptr) {
        ssize_t threads = threads_ptr == nullptr ? std::thread::hardware_concurrency() : *threads_ptr;
        size_t n        = dim;

        std::vector<std::thread> threadpool;
        std::vector<fill_helper<Histogram>> helpers;

        for(ssize_t i = 0; i < threads; i++) {
            helpers.emplace_back(make_threaded(threads, i));
        }

        {
            py::gil_scoped_release tmp;
            for(auto &helper : helpers) {
                threadpool.emplace_back([&helper, n]() {
                    boost::mp11::mp_with_index<BOOST_HISTOGRAM_DETAIL_AXES_LIMIT>(n, std::ref(helper));
                });
            }
        }

        for(auto &thread : threadpool)
            thread.join();

        for(auto &helper : helpers) {
            auto &s  = bh::unsafe_access::storage(*hist);
            auto rit = bh::unsafe_access::storage(*helper.hist).begin();
            std::for_each(s.begin(), s.end(), [&rit](auto &&x) { x += *rit++; });
        }
    }

    // Atomic fill
    void fill_atomic(ssize_t * atomic_ptr) {
        ssize_t threads = atomic_ptr == nullptr ? std::thread::hardware_concurrency() : *atomic_ptr;
        size_t n        = dim;

        std::vector<std::thread> threadpool;
        std::vector<fill_helper<Histogram>> helpers;

        for(ssize_t i = 0; i < threads; i++) {
            helpers.emplace_back(make_atomic(threads, i));
        }

        {
            py::gil_scoped_release tmp;
            for(auto &helper : helpers) {
                threadpool.emplace_back([&helper, n]() {
                    boost::mp11::mp_with_index<BOOST_HISTOGRAM_DETAIL_AXES_LIMIT>(n, std::ref(helper));
                });
            }
        }

        for(auto &thread : threadpool)
            thread.join();
    }
};

template <class Histogram>
struct [[gnu::visibility("hidden")]] fill_helper_weights {
    Histogram *hist;
    std::shared_ptr<Histogram> local_hist;        // Only used when threading - empty otherwise
    std::vector<py::array_t<double>> source_ptrs; // Stores a copy of converted values *if* conversion is neccesary
    std::vector<const double *> data_ptrs;        // The actually data is stored here
    ssize_t size = 1;                             // How many values to copy
    size_t dim;                                   // Current number of dimensions to fill
    const double *weights;                        // Weights array pointer

    fill_helper_weights(Histogram & h, py::array_t<double> & weights_arr, py::args args)
        : hist(&h)
        , weights(weights_arr.data()) {
        dim = args.size();
        if(dim == 0)
            throw std::invalid_argument("at least one argument required");

        source_ptrs.reserve(dim);
        data_ptrs.reserve(dim);

        for(size_t i = 0; i < dim; ++i) {
            auto a = py::cast<py::array_t<double>>(args[i]);

            if(a.ndim() > 1)
                throw std::invalid_argument("array dim must be 0 or 1");

            if(size != 1 && size != a.size())
                throw std::invalid_argument("arrays must have same length");
            size = a.size();

            // We keep the python objects just in case a conversion was made
            // and the Python reference counting is needed.
            source_ptrs.emplace_back(a);

            data_ptrs.emplace_back(a.data());
        }

        if(weights_arr.size() != size)
            throw std::invalid_argument("weight array must have same length as data");
    }

    /// Make an atomic copy (same histogram internally)
    fill_helper_weights<Histogram> make_atomic(ssize_t total_threads, ssize_t current_thread) {
        fill_helper_weights<Histogram> self{*this};

        ssize_t start = current_thread * size / total_threads;
        ssize_t stop  = (current_thread + 1 == total_threads) ? size : (current_thread + 1) * size / total_threads;
        self.size     = stop - start; // Start is always less than or equal to stop

        for(auto &item : self.data_ptrs) {
            item += start;
        }

        return self;
    }

    /// Make a threaded copy (new histogram internally)
    fill_helper_weights<Histogram> make_threaded(ssize_t total_threads, ssize_t current_thread) {
        fill_helper_weights<Histogram> self = make_atomic(total_threads, current_thread);

        // Make a new "optional" local histogram, and make the hist pointer point at it instead of the master
        self.local_hist.reset(new Histogram(*hist));
        self.hist = self.local_hist.get();

        return self;
    }

    // Standard fill
    void fill() {
        py::gil_scoped_release tmp;
        boost::mp11::mp_with_index<BOOST_HISTOGRAM_DETAIL_AXES_LIMIT>(dim, std::ref(*this));
    }

    // Threaded fill
    void fill_threaded(ssize_t * threads_ptr) {
        ssize_t threads = threads_ptr == nullptr ? std::thread::hardware_concurrency() : *threads_ptr;
        size_t n        = dim;

        std::vector<std::thread> threadpool;
        std::vector<fill_helper_weights<Histogram>> helpers;

        for(ssize_t i = 0; i < threads; i++) {
            helpers.emplace_back(make_threaded(threads, i));
        }

        {
            py::gil_scoped_release tmp;
            for(auto &helper : helpers) {
                threadpool.emplace_back([&helper, n]() {
                    boost::mp11::mp_with_index<BOOST_HISTOGRAM_DETAIL_AXES_LIMIT>(n, std::ref(helper));
                });
            }
        }

        for(auto &thread : threadpool)
            thread.join();

        for(auto &helper : helpers) {
            auto &s  = bh::unsafe_access::storage(*hist);
            auto rit = bh::unsafe_access::storage(*helper.hist).begin();
            std::for_each(s.begin(), s.end(), [&rit](auto &&x) { x += *rit++; });
        }
    }

    // Atomic fill
    void fill_atomic(ssize_t * atomic_ptr) {
        ssize_t threads = atomic_ptr == nullptr ? std::thread::hardware_concurrency() : *atomic_ptr;
        size_t n        = dim;

        std::vector<std::thread> threadpool;
        std::vector<fill_helper_weights<Histogram>> helpers;

        for(ssize_t i = 0; i < threads; i++) {
            helpers.emplace_back(make_atomic(threads, i));
        }

        {
            py::gil_scoped_release tmp;
            for(auto &helper : helpers) {
                threadpool.emplace_back([&helper, n]() {
                    boost::mp11::mp_with_index<BOOST_HISTOGRAM_DETAIL_AXES_LIMIT>(n, std::ref(helper));
                });
            }
        }

        for(auto &thread : threadpool)
            thread.join();
    }

    // keep function small to minimize code bloat, it is instantiated 16 times :(
    // TODO: solve this more efficiently on the lower Boost::Histogram level
    template <class N>
    void operator()(N) {
        using namespace boost::mp11;

        // N is a compile-time number with N == arrs.size()
        // Type: tuple<double, ..., double> (N times)
        mp_append<mp_repeat<std::tuple<double>, N>, mp_list<bh::weight_type<double>>> tp;
        using m_list         = mp_iota<N>; // list of integral constants
        const double *weight = weights;

        for(std::size_t i = 0; i < (std::size_t)size; ++i) {
            mp_for_each<m_list>([&](auto I) {
                // I is mp_size_t<0>, mp_size_t<1>, ..., mp_size_t<N-1>
                std::get<I>(tp) = data_ptrs[I][i];
            });
            double weighted_value  = *weight++;
            std::get<N::value>(tp) = bh::weight(std::move(weighted_value));

            (*hist)(tp); // throws invalid_argument if tup has wrong size
        }
    }

    // specialization for N=0 to prevent compile-time error in histogram
    void operator()(boost::mp11::mp_size_t<0>) { throw std::invalid_argument("at least one argument required"); }

    // specialization for N=1, implement potential optimizations here
    void operator()(boost::mp11::mp_size_t<1>) {
        const double *data   = this->data_ptrs[0];
        const double *end    = this->data_ptrs[0] + size;
        const double *weight = weights;

        for(; data != end; data++, weight++)
            (*hist)(*data, bh::weight(*weight)); // throws invalid_argument if hist.rank() != 1
    }
};
