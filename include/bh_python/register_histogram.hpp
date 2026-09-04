// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <bh_python/pybind11.hpp>

#include <bh_python/accumulators/ostream.hpp>
#include <bh_python/axis.hpp>
#include <bh_python/def_eq.hpp>
#include <bh_python/fill.hpp>
#include <bh_python/histogram.hpp>
#include <bh_python/make_pickle.hpp>
#include <bh_python/storage.hpp>

#include <boost/histogram/algorithm/empty.hpp>
#include <boost/histogram/algorithm/project.hpp>
#include <boost/histogram/algorithm/reduce.hpp>
#include <boost/histogram/algorithm/sum.hpp>
#include <boost/histogram/histogram.hpp>
#include <boost/histogram/indexed.hpp>
#include <boost/histogram/ostream.hpp>
#include <boost/histogram/unsafe_access.hpp>
#include <boost/mp11.hpp>

#include <algorithm>
#include <future>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <vector>

/// A growth axis is replaced or reallocated by a growing fill or merge, so a
/// reference to it would dangle. Give a copy instead; its bins are a snapshot.
/// Other axes stay no-copy, held alive by py::keep_alive on the histogram.
template <class A>
py::object growth_safe_axis_cast(const A& item, std::true_type /*growing*/) {
    py::object obj = py::cast(item, py::return_value_policy::copy);
    // The copy stands in for the stored axis, so it must hold the same
    // metadata dict, not the independent one that copying a metadata_t makes.
    py::cast<A&>(obj).metadata()
        = metadata_t{py::object(item.metadata().unguarded_obj())};
    return obj;
}

template <class A>
py::object growth_safe_axis_cast(const A& item, std::false_type /*growing*/) {
    return py::cast(item, py::return_value_policy::reference);
}

template <class A>
using axis_is_growing = std::integral_constant<bool,
                                               bh::axis::traits::get_options<A>::test(
                                                   bh::axis::option::growth)>;

/// In-place operator functors. The member template is only instantiated by
/// the std::true_type def_inplace_op, so a storage without the operator
/// still compiles.
struct op_iadd {
    template <class H>
    void operator()(H& a, const H& b) const {
        a += b;
    }
};
struct op_isub {
    template <class H>
    void operator()(H& a, const H& b) const {
        a -= b;
    }
};
struct op_imul {
    template <class H>
    void operator()(H& a, const H& b) const {
        a *= b;
    }
};
struct op_itruediv {
    template <class H>
    void operator()(H& a, const H& b) const {
        a /= b;
    }
};

/// Define an in-place histogram operator that runs without the GIL and gives
/// back the same Python object (a registered instance is returned as is).
template <class H, class... Extra, class Op>
void def_inplace_op(py::class_<H, Extra...>& hist,
                    std::true_type,
                    const char* name,
                    Op) {
    hist.def(
        name,
        [](H& self, const H& other) -> H& {
            const py::gil_scoped_release release;
            Op{}(self, other);
            return self;
        },
        py::is_operator(),
        py::return_value_policy::reference);
}

template <class H, class... Extra, class Op>
void def_inplace_op(py::class_<H, Extra...>&, std::false_type, const char*, Op) {}

/// Whole-buffer operations run with the GIL released. The storage is plain
/// C++; the axes hold metadata through guarded_object, which takes the GIL
/// itself for the reference counting a copy or compare needs.
template <class H, class... Extra>
void def_buffer_ops(py::class_<H, Extra...>& hist) {
    def_eq<py::gil_scoped_release>(hist);

    hist.def("reset",
             [](H& self) {
                 const py::gil_scoped_release release;
                 self.reset();
             })
        .def("__copy__",
             [](const H& self) {
                 const py::gil_scoped_release release;
                 return H(self);
             })
        .def("__deepcopy__", [](const H& self, const py::object& memo) {
            auto a = [&self] {
                const py::gil_scoped_release release;
                return std::make_unique<H>(self);
            }();
            for(unsigned i = 0; i < a->rank(); i++) {
                bh::unsafe_access::axis(*a, i).metadata()
                    = deep_copy_metadata(a->axis(i).metadata(), memo);
            }
            return a;
        });

    def_inplace_op(hist, std::true_type{}, "__iadd__", op_iadd{});
    def_inplace_op(hist, bh::detail::has_operator_rsub<H, H>{}, "__isub__", op_isub{});
    def_inplace_op(hist, bh::detail::has_operator_rmul<H, H>{}, "__imul__", op_imul{});
    def_inplace_op(
        hist, bh::detail::has_operator_rdiv<H, H>{}, "__itruediv__", op_itruediv{});
}

template <class S>
auto register_histogram(py::module& m, const char* name, const char* desc) {
    using histogram_t = bh::histogram<vector_axis_variant, S>;
    using value_type  = typename histogram_t::value_type;

    py::class_<histogram_t> hist(m, name, desc, py::buffer_protocol());
    def_buffer_ops(hist);

    hist.def(py::init<const vector_axis_variant&, S>(), "axes"_a, "storage"_a = S())

        .def_buffer(
            [](histogram_t& h) -> py::buffer_info { return make_buffer(h, false); })

        .def("rank", &histogram_t::rank)
        .def("size", &histogram_t::size)

        .def_property_readonly_static(
            "_storage_type",
            [](const py::object&) {
                return py::type::of<typename histogram_t::storage_type>();
            })

        .def(
            "to_numpy",
            [](histogram_t& h, bool flow) {
                py::tuple tup(1 + h.rank());

                // Add the histogram buffer as the first argument
                unchecked_set(tup, 0, py::array(make_buffer(h, flow)));

                // Add the axis edges
                h.for_each_axis([&tup, flow, i = 0U](const auto& ax) mutable {
                    unchecked_set(tup, ++i, axis::edges(ax, flow, true));
                });

                return tup;
            },
            "flow"_a = false)

        .def(
            "view",
            [](const py::object& self, bool flow) {
                auto& h = py::cast<histogram_t&>(self);
                return py::array(make_buffer(h, flow), self);
            },
            "flow"_a = false)

        .def(
            "axis",
            [](const histogram_t& self, int i) -> py::object {
                unsigned const ii
                    = i < 0 ? self.rank() - static_cast<unsigned>(std::abs(i))
                            : static_cast<unsigned>(i);

                if(ii < self.rank()) {
                    const axis_variant& var = self.axis(ii);
                    return bh::axis::visit(
                        [](auto&& item) -> py::object {
                            using item_t = std::decay_t<decltype(item)>;
                            return growth_safe_axis_cast(item,
                                                         axis_is_growing<item_t>{});
                        },
                        var);
                }

                throw std::out_of_range("The axis value must be less than the rank");
            },
            "i"_a = 0,
            py::keep_alive<0, 1>())

        // Setting metadata through axis() would only reach the copy handed out
        // for a growth axis, so set it on the stored axis instead.
        .def("_set_axis_metadata",
             [](histogram_t& self, unsigned i, metadata_t data) {
                 if(i >= self.rank())
                     throw std::out_of_range(
                         "The axis value must be less than the rank");
                 bh::unsafe_access::axis(self, i).metadata() = std::move(data);
             })

        .def("at",
             [](const histogram_t& self, const py::args& args) -> value_type {
                 auto int_args = py::cast<std::vector<int>>(args);
                 return self.at(int_args);
             })

        .def("_at_set",
             [](histogram_t& self, const value_type& input, const py::args& args) {
                 auto int_args     = py::cast<std::vector<int>>(args);
                 self.at(int_args) = input;
             })

        .def("__repr__", &shift_to_string<histogram_t>)

        .def(
            "sum",
            [](const histogram_t& self, bool flow) {
                const py::gil_scoped_release release;
                // A rank-0 histogram has no flow bins, so inner == all. Use
                // all to avoid Boost's indexed range, which is UB for rank-0
                // (it reads uninitialized per-axis state); the all path uses
                // plain iteration instead.
                const auto cov = (flow || self.rank() == 0) ? bh::coverage::all
                                                            : bh::coverage::inner;
                return bh::algorithm::sum(self, cov);
            },
            "flow"_a = false)

        .def(
            "empty",
            [](const histogram_t& self, bool flow) {
                const py::gil_scoped_release release;
                if(self.rank() == 0) {
                    // algorithm::empty drives the same rank-0-UB indexed range;
                    // check the single cell directly instead.
                    using value_type = typename histogram_t::value_type;
                    return !(*self.begin() != value_type());
                }
                return bh::algorithm::empty(
                    self, flow ? bh::coverage::all : bh::coverage::inner);
            },
            "flow"_a = false)

        .def("reduce",
             [](const histogram_t& self, const py::args& args) -> histogram_t {
                 auto commands
                     = py::cast<std::vector<bh::algorithm::reduce_command>>(args);
                 // reduce drives the same rank-0-UB indexed range; with no
                 // commands there is nothing to do, and any axis index is
                 // rejected before that point.
                 if(self.rank() == 0 && commands.empty())
                     return self;
                 const py::gil_scoped_release release;
                 return bh::algorithm::reduce(self, commands);
             })

        .def("project",
             [](const histogram_t& self, const py::args& values) -> histogram_t {
                 auto cpp_values = py::cast<std::vector<unsigned>>(values);
                 // Same rank-0-UB indexed range; the identity is the only
                 // projection a rank-0 histogram has.
                 if(self.rank() == 0 && cpp_values.empty())
                     return self;
                 const py::gil_scoped_release release;
                 return bh::algorithm::project(self, cpp_values);
             })

        .def("fill", &fill<histogram_t>)

        .def(make_pickle<histogram_t>())

        ;

    return hist;
}

template <>
auto inline register_histogram<bh::multi_cell<double>>(py::module& m,
                                                       const char* name,
                                                       const char* desc) {
    using S           = bh::multi_cell<double>;
    using histogram_t = bh::histogram<vector_axis_variant, S>;
    using value_type  = std::vector<double>;

    py::class_<histogram_t> hist(m, name, desc, py::buffer_protocol());
    def_buffer_ops(hist);

    hist.def(py::init<const vector_axis_variant&, S>(), "axes"_a, "storage"_a = S())

        .def_buffer(
            [](histogram_t& h) -> py::buffer_info { return make_buffer(h, false); })

        .def("rank", &histogram_t::rank)
        .def("size", &histogram_t::size)
        .def("nelem",
             [](const histogram_t& self) {
                 return bh::unsafe_access::storage(self).nelem();
             })

        // Reset number of cells per bin after recreation of histogram because number
        // of cells can (?) not be passed to the creation of the new histogram. Set it
        // manually afterwards.
        .def("reset_nelem",
             [](histogram_t& self, const std::size_t nelem) {
                 bh::unsafe_access::storage(self).reset_nelem(nelem);
             })

        .def_property_readonly_static(
            "_storage_type",
            [](const py::object&) {
                return py::type::of<typename histogram_t::storage_type>();
            })

        .def(
            "to_numpy",
            [](histogram_t& h, bool flow) {
                py::tuple tup(1 + h.rank());

                // Add the histogram buffer as the first argument
                unchecked_set(tup, 0, py::array(make_buffer(h, flow)));

                // Add the axis edges
                h.for_each_axis([&tup, flow, i = 0U](const auto& ax) mutable {
                    unchecked_set(tup, ++i, axis::edges(ax, flow, true));
                });

                return tup;
            },
            "flow"_a = false)

        .def(
            "view",
            [](const py::object& self, bool flow) {
                auto& h = py::cast<histogram_t&>(self);
                return py::array(make_buffer(h, flow), self);
            },
            "flow"_a = false)

        .def(
            "axis",
            [](const histogram_t& self, int i) -> py::object {
                unsigned const ii
                    = i < 0 ? self.rank() - static_cast<unsigned>(std::abs(i))
                            : static_cast<unsigned>(i);

                if(ii < self.rank()) {
                    const axis_variant& var = self.axis(ii);
                    return bh::axis::visit(
                        [](auto&& item) -> py::object {
                            using item_t = std::decay_t<decltype(item)>;
                            return growth_safe_axis_cast(item,
                                                         axis_is_growing<item_t>{});
                        },
                        var);
                }

                throw std::out_of_range("The axis value must be less than the rank");
            },
            "i"_a = 0,
            py::keep_alive<0, 1>())

        // Setting metadata through axis() would only reach the copy handed out
        // for a growth axis, so set it on the stored axis instead.
        .def("_set_axis_metadata",
             [](histogram_t& self, unsigned i, metadata_t data) {
                 if(i >= self.rank())
                     throw std::out_of_range(
                         "The axis value must be less than the rank");
                 bh::unsafe_access::axis(self, i).metadata() = std::move(data);
             })

        .def("at",
             [](const histogram_t& self, const py::args& args) -> value_type {
                 auto int_args = py::cast<std::vector<int>>(args);
                 auto at_value = self.at(int_args);
                 return {at_value.begin(), at_value.end()};
             })

        .def("_at_set",
             [](histogram_t& self, const value_type& input, const py::args& args) {
                 auto int_args     = py::cast<std::vector<int>>(args);
                 self.at(int_args) = input;
             })

        .def("__repr__", &shift_to_string<histogram_t>)

        .def(
            "sum",
            [](const histogram_t& self, bool flow) -> value_type {
                const py::gil_scoped_release release;
                // rank-0 inner coverage drives Boost's indexed range, which is
                // UB for rank-0; all coverage is equivalent (no flow bins) and
                // uses plain iteration.
                const auto cov    = (flow || self.rank() == 0) ? bh::coverage::all
                                                               : bh::coverage::inner;
                value_type result = bh::algorithm::sum(self, cov);
                // A histogram with zero bins has no cells to accumulate, so the
                // default-constructed accumulator stays empty. Return a
                // zero-filled vector of length nelem so the result shape is
                // consistent with the non-empty case.
                if(result.empty()) {
                    result.assign(bh::unsafe_access::storage(self).nelem(), 0.0);
                }
                return result;
            },
            "flow"_a = false)

        .def(
            "empty",
            [](const histogram_t& self, bool flow) {
                const py::gil_scoped_release release;
                // A cell is empty when every element is zero; algorithm::empty
                // compares against a default-constructed (zero length) cell,
                // which never matches. rank 0 also drives the indexed range,
                // which is UB for rank-0 histograms.
                const auto all_zero = [](const auto& cell) {
                    return std::all_of(
                        cell.begin(), cell.end(), [](double x) { return x == 0; });
                };
                if(flow || self.rank() == 0)
                    return std::all_of(self.begin(), self.end(), all_zero);
                auto range = bh::indexed(self);
                return std::all_of(range.begin(),
                                   range.end(),
                                   [&all_zero](const auto& x) { return all_zero(*x); });
            },
            "flow"_a = false)

        .def("reduce",
             [](const histogram_t& self, const py::args& args) -> histogram_t {
                 auto commands
                     = py::cast<std::vector<bh::algorithm::reduce_command>>(args);
                 // reduce drives the same rank-0-UB indexed range; with no
                 // commands there is nothing to do, and any axis index is
                 // rejected before that point.
                 if(self.rank() == 0 && commands.empty())
                     return self;
                 const py::gil_scoped_release release;
                 return bh::algorithm::reduce(self, commands);
             })

        .def("project",
             [](const histogram_t& self, const py::args& values) -> histogram_t {
                 auto cpp_values = py::cast<std::vector<unsigned>>(values);
                 // Same rank-0-UB indexed range; the identity is the only
                 // projection a rank-0 histogram has.
                 if(self.rank() == 0 && cpp_values.empty())
                     return self;
                 const py::gil_scoped_release release;
                 return bh::algorithm::project(self, cpp_values);
             })

        .def("fill", &fill<histogram_t>)

        .def(make_pickle<histogram_t>())

        ;

    return hist;
}
