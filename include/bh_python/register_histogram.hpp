// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <bh_python/pybind11.hpp>

#include <bh_python/accumulators/ostream.hpp>
#include <bh_python/axis.hpp>
#include <bh_python/fill.hpp>
#include <bh_python/histogram.hpp>
#include <bh_python/make_pickle.hpp>
#include <bh_python/storage.hpp>

#include <boost/histogram/algorithm/empty.hpp>
#include <boost/histogram/algorithm/project.hpp>
#include <boost/histogram/algorithm/reduce.hpp>
#include <boost/histogram/algorithm/sum.hpp>
#include <boost/histogram/histogram.hpp>
#include <boost/histogram/ostream.hpp>
#include <boost/histogram/unsafe_access.hpp>
#include <boost/mp11.hpp>

#include <future>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

template <class S>
auto register_histogram(py::module& m, const char* name, const char* desc) {
    using histogram_t = bh::histogram<vector_axis_variant, S>;
    using value_type  = typename histogram_t::value_type;

    py::class_<histogram_t> hist(m, name, desc, py::buffer_protocol());

    hist.def(py::init<const vector_axis_variant&, S>(), "axes"_a, "storage"_a = S())

        .def_buffer(
            [](histogram_t& h) -> py::buffer_info { return make_buffer(h, false); })

        .def("rank", &histogram_t::rank)
        .def("size", &histogram_t::size)
        .def("reset", &histogram_t::reset)

        .def("__copy__", [](const histogram_t& self) { return histogram_t(self); })
        .def("__deepcopy__",
             [](const histogram_t& self, py::object memo) {
                 auto* a         = new histogram_t(self);
                 py::module copy = py::module::import("copy");
                 for(unsigned i = 0; i < a->rank(); i++) {
                     bh::unsafe_access::axis(*a, i).metadata()
                         = copy.attr("deepcopy")(a->axis(i).metadata(), memo);
                 }
                 return a;
             })

        .def(py::self += py::self)

        .def("__eq__",
             [](const histogram_t& self, const py::object& other) {
                 try {
                     return self == py::cast<histogram_t>(other);
                 } catch(const py::cast_error&) {
                     return false;
                 }
             })
        .def("__ne__",
             [](const histogram_t& self, const py::object& other) {
                 try {
                     return self != py::cast<histogram_t>(other);
                 } catch(const py::cast_error&) {
                     return true;
                 }
             })

        .def_property_readonly_static(
            "_storage_type",
            [](py::object) {
                return py::type::of<typename histogram_t::storage_type>();
            })

        ;

// Protection against an overzealous warning system
// https://bugs.llvm.org/show_bug.cgi?id=43124
#ifdef __clang__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wself-assign-overloaded"
#endif
    def_optionally(hist,
                   bh::detail::has_operator_rdiv<histogram_t, histogram_t>{},
                   py::self /= py::self);
    def_optionally(hist,
                   bh::detail::has_operator_rmul<histogram_t, histogram_t>{},
                   py::self *= py::self);
#ifdef __clang__
#pragma GCC diagnostic pop
#endif

    hist.def(
            "to_numpy",
            [](histogram_t& h, bool flow) {
                py::tuple tup(1 + h.rank());

                // Add the histogram buffer as the first argument
                unchecked_set(tup, 0, py::array(make_buffer(h, flow)));

                // Add the axis edges
                h.for_each_axis([&tup, flow, i = 0u](const auto& ax) mutable {
                    unchecked_set(tup, ++i, axis::edges(ax, flow, true));
                });

                return tup;
            },
            "flow"_a = false)

        .def(
            "view",
            [](py::object self, bool flow) {
                auto& h = py::cast<histogram_t&>(self);
                return py::array(make_buffer(h, flow), self);
            },
            "flow"_a = false)

        .def(
            "axis",
            [](const histogram_t& self, int i) -> py::object {
                unsigned ii = i < 0 ? self.rank() - static_cast<unsigned>(std::abs(i))
                                    : static_cast<unsigned>(i);

                if(ii < self.rank()) {
                    const axis_variant& var = self.axis(ii);
                    return bh::axis::visit(
                        [](auto&& item) -> py::object {
                            // Here we return a new, no-copy py::object that
                            // is not yet tied to the histogram. py::keep_alive
                            // is needed to make sure the histogram is alive as long
                            // as the axes references are.
                            return py::cast(item, py::return_value_policy::reference);
                        },
                        var);

                }

                else
                    throw std::out_of_range(
                        "The axis value must be less than the rank");
            },
            "i"_a = 0,
            py::keep_alive<0, 1>())

        .def("at",
             [](const histogram_t& self, py::args& args) -> value_type {
                 auto int_args = py::cast<std::vector<int>>(args);
                 return self.at(int_args);
             })

        .def("_at_set",
             [](histogram_t& self, const value_type& input, py::args& args) {
                 auto int_args     = py::cast<std::vector<int>>(args);
                 self.at(int_args) = input;
             })

        .def("__repr__", &shift_to_string<histogram_t>)

        .def(
            "sum",
            [](const histogram_t& self, bool flow) {
                return bh::algorithm::sum(
                    self, flow ? bh::coverage::all : bh::coverage::inner);
            },
            "flow"_a = false)

        .def(
            "empty",
            [](const histogram_t& self, bool flow) {
                return bh::algorithm::empty(
                    self, flow ? bh::coverage::all : bh::coverage::inner);
            },
            "flow"_a = false)

        .def("reduce",
             [](const histogram_t& self, py::args args) {
                 return bh::algorithm::reduce(
                     self, py::cast<std::vector<bh::algorithm::reduce_command>>(args));
             })

        .def("project",
             [](const histogram_t& self, py::args values) {
                 return bh::algorithm::project(self,
                                               py::cast<std::vector<unsigned>>(values));
             })

        .def("fill", &fill<histogram_t>)

        .def(make_pickle<histogram_t>())

        ;

    return hist;
}
