// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <boost/histogram/python/pybind11.hpp>

#include <boost/histogram/indexed.hpp>
#include <pybind11/numpy.h>

#include <type_traits>
#include <vector>

template <class T>
struct index_python {
    using value_ref_t = typename decltype(std::declval<T>().begin())::reference;

    T &histogram_;
    std::array<bh::axis::index_type, BOOST_HISTOGRAM_DETAIL_AXES_LIMIT> indices_;
    value_ref_t content;

    template <class It>
    index_python(T &histogram, value_ref_t content, It &&begin_it, It &&end_it)
        : histogram_(histogram)
        , content(content) {
        if(histogram_.rank() > indices_.size())
            throw std::invalid_argument("histogram rank too large");
        std::copy(begin_it, end_it, indices_.begin());
    }

    unsigned rank() const { return histogram_.rank(); }

    py::tuple bins() const {
        py::tuple tup(rank());
        histogram_.for_each_axis([&tup, this, i = 0u](const auto &ax) mutable {
            unchecked_set(tup, i, axis::unchecked_bin(ax, indices_[i]));
            ++i;
        });
        return tup;
    }

    py::tuple indices() const {
        py::tuple tup(rank());
        for(unsigned i = 0; i < rank(); ++i)
            unchecked_set(tup, i, indices_[i]);
        return tup;
    }

    const value_ref_t get_content() const { return content; }

    value_ref_t get_content() { return content; }

    py::tuple centers() {
        py::tuple tup(rank());
        histogram_.for_each_axis([&tup, this, i = 0u](const auto &ax) mutable {
            unchecked_set(tup, i, axis::unchecked_center(ax, indices_[i]));
            ++i;
        });
        return tup;
    }

    // Add content(s), centers, edges
};

template <class histogram_t>
struct repeatable_indexed {
    histogram_t &histogram;
    bh::coverage cov;
    using indexed_t = decltype(
        bh::indexed(std::declval<histogram_t &>(), std::declval<bh::coverage>()));
    using iterator_t = decltype(std::declval<indexed_t>().begin());
    std::unique_ptr<indexed_t> ind;
    iterator_t it;
    iterator_t end;
    ssize_t idx{0};

    repeatable_indexed(const repeatable_indexed &) = default;
    repeatable_indexed(repeatable_indexed &&)      = default;

    repeatable_indexed(histogram_t &histogram, bh::coverage cov)
        : histogram(histogram)
        , cov(cov)
        , ind(std::make_unique<indexed_t>(histogram, cov))
        , it(ind->begin())
        , end(ind->end()) {}

    void restart() {
        ind.reset(new indexed_t{histogram, cov});

        it  = ind->begin();
        end = ind->end();
        idx = 0;
    }

    repeatable_indexed &operator++() {
        ++it;
        ++idx;
        return *this;
    }

    index_python<histogram_t> get() {
        auto result = it->indices();
        return index_python<histogram_t>(
            histogram,
            it->get(),
            result.begin(),
            result.end()); // Copy elision, for sure in C++17
    }
};

/// Makes a python iterator from a first and past-the-end C++ InputIterator.
template <py::return_value_policy Policy = py::return_value_policy::reference_internal,
          typename histogram_ref_t,
          typename... Extra>
py::iterator make_repeatable_iterator(histogram_ref_t &histogram,
                                      bh::coverage cov,
                                      Extra &&... extra) {
    using histogram_t = std::decay_t<histogram_ref_t>;
    using state       = repeatable_indexed<histogram_t>;

    if(!py::detail::get_type_info(typeid(state), false)) {
        py::class_<state>(py::handle(), "iterator", py::module_local())
            .def(
                "__iter__",
                [](state &s) -> state & {
                    s.restart();
                    return s;
                },
                py::keep_alive<0, 1>())
            .def(
                "__next__",
                [](state &s) -> index_python<histogram_t> {
                    if(s.it == s.end)
                        throw py::stop_iteration();

                    index_python<histogram_t> ip = s.get();
                    ++s;
                    return ip;
                },
                std::forward<Extra>(extra)...,
                Policy);
    }

    return py::cast(state{histogram, cov}, py::return_value_policy::move);
}

template <class histogram_t>
void register_indexed(py::module &m, std::string name) {
    using indexed_t = index_python<histogram_t>;

    std::string name_indexed = name + "_indexed";

    py::class_<indexed_t> indexed{m, name_indexed.c_str()};

    indexed.def("bins", &indexed_t::bins);
    indexed.def("indices", &indexed_t::indices);
    indexed.def("centers", &indexed_t::centers);

    indexed.def_property(
        "content",
        [](const indexed_t &self) { return self.get_content(); },
        [](indexed_t &self, typename histogram_t::value_type value) {
            self.get_content() = value;
        });
}

template <class histogram_t>
void register_ufunc_tools(py::class_<histogram_t> &cls) {
    cls.def(
        "centers",
        [](const histogram_t &hist, bool flow) {
            using array_t = py::array_t<double>;
            array_t::ShapeContainer shapes;
            for(unsigned i = 0; i < hist.rank(); i++) {
                shapes->push_back(flow ? bh::axis::traits::extent(hist.axis(i))
                                       : hist.axis(i).size());
            }

            std::vector<double> centers;
            py::array_t<double> arr;
            arr.resize(shapes);

            for(auto &&ind :
                bh::indexed(hist, flow ? bh::coverage::all : bh::coverage::inner)) {
                ssize_t tot = 0;
                for(size_t i = 0; i < hist.rank(); i++) {
                    tot += ind.index((unsigned)i) * 1; // TODO: Support ND
                }
                *(arr.mutable_data() + tot) = ind.bin(0).center();
            }

            return arr;
        },
        "flow"_a = false);
}
