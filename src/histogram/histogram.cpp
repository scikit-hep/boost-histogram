// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#include <boost/histogram/python/pybind11.hpp>
#include <pybind11/operators.h>

#include <boost/histogram/python/axis.hpp>
#include <boost/histogram/python/storage.hpp>
#include <boost/histogram/python/histogram.hpp>
#include <boost/histogram/python/histogram_fill.hpp>

#include <boost/histogram.hpp>
#include <boost/histogram/axis/ostream.hpp>
#include <boost/histogram/ostream.hpp>

#include <boost/mp11.hpp>

#include <cassert>
#include <vector>
#include <sstream>
#include <tuple>
#include <cmath>

namespace bh = boost::histogram;


template<typename A, typename S>
py::class_<bh::histogram<A, S>> register_histogram_by_type(py::module& m, const char* name, const char* desc) {

    using histogram_t = bh::histogram<A, S>;

    py::class_<histogram_t> hist(m, name, desc, py::buffer_protocol());

    hist
    .def(py::init<const A&, S>(), "axes"_a, "storage"_a=S())

    .def_buffer([](bh::histogram<A, S>& h) -> py::buffer_info
        {return make_buffer(h, true);})

    .def("rank", &histogram_t::rank,
         "Number of axes (dimensions) of histogram" )
    .def("size", &histogram_t::size,
         "Total number of bins in the histogram (including underflow/overflow)" )
    .def("reset", &histogram_t::reset,
         "Reset bin counters to zero")

    .def(py::self + py::self)
    .def(py::self == py::self)
    .def(py::self != py::self)
    .def(py::self *= double())
    .def(py::self /= double())

    .def("to_numpy", [](histogram_t& h, bool flow){
        py::list listing;

        // Add the histogram as the first argument
        py::array arr(make_buffer(h, flow));
        listing.append(arr);

        // Add the axis edges
        for(unsigned i=0; i<h.rank(); i++) {
            const auto& ax = h.axis(i);
            listing.append(axis_to_edges(ax, flow));
        }

        return py::cast<py::tuple>(listing);
    },
         "flow"_a = false, "convert to a numpy style tuple of returns")

    .def("axis",
        [](histogram_t &self, unsigned i){return self.axis(i);},
     "Get N-th axis with runtime index",
         py::return_value_policy::move)

    // generic fill for 1 to N args
    .def("__call__",
      [](histogram_t &self, py::args args) {
        boost::mp11::mp_with_index<BOOST_HISTOGRAM_DETAIL_AXES_LIMIT>(args.size(), fill_helper<histogram_t>(self, args));
      }, "Insert data into histogram")

    .def("at", [](histogram_t &self, py::args &args) {
        // Optimize for no dynamic?
        auto int_args = py::cast<std::vector<int>>(args);
        return self.at(int_args);
        },
         "Access bin counter at indices")

    .def("__repr__", [](histogram_t &self){
        std::ostringstream out;
        out << self;
        return out.str();
    })

    ;

    return hist;
}

void register_histogram(py::module& m) {
    py::module hist = m.def_submodule("hist");

    // Fast specializations: Fixed number of axis (may be removed if above versions are fast enough)
    // Mostly targeting histogram styles supported by numpy for these max performance versions.

    register_histogram_by_type<axes::regular_1D, dense_int_storage>(hist,
         "regular_int_1d",
         "1-dimensional histogram for int valued data.");

    m.def("make_histogram", [](axis::regular& ax1, dense_int_storage){
        return bh::make_histogram_with(dense_int_storage(), ax1);
    }, "axis"_a, "storage"_a=dense_int_storage(), "Make a 1D histogram of integers");


    register_histogram_by_type<axes::regular_2D, dense_int_storage>(hist,
        "regular_int_2d",
        "2-dimensional histogram for int valued data.");

    m.def("make_histogram", [](axis::regular& ax1, axis::regular& ax2, dense_int_storage){
        return bh::make_histogram_with(dense_int_storage(), ax1, ax2);
    }, "axis1"_a, "axis2"_a, "storage"_a=dense_int_storage(), "Make a 2D histogram of integers");


    register_histogram_by_type<axes::regular_noflow_1D, dense_int_storage>(hist,
        "regular_int_noflow_1d",
        "1-dimensional histogram for int valued data.");

    m.def("make_histogram", [](axis::regular_noflow& ax1, dense_int_storage){
        return bh::make_histogram_with(dense_int_storage(), ax1);
    }, "axis"_a, "storage"_a=dense_int_storage(), "Make a 1D histogram of integers without overflow");


    register_histogram_by_type<axes::regular_noflow_2D, dense_int_storage>(hist,
        "regular_int_noflow_2d",
        "2-dimensional histogram for int valued data.");

    m.def("make_histogram", [](axis::regular_noflow& ax1, axis::regular_noflow& ax2, dense_int_storage){
        return bh::make_histogram_with(dense_int_storage(), ax1, ax2);
    }, "axis1"_a, "axis2"_a, "storage"_a=dense_int_storage(), "Make a 2D histogram of integers without overflow");


    // Fast specializations - uniform types

    register_histogram_by_type<axes::regular, bh::unlimited_storage<>>(hist,
        "regular_unlimited",
        "N-dimensional histogram for real-valued data.");

    register_histogram_by_type<axes::regular, dense_int_storage>(hist,
        "regular_int",
        "N-dimensional histogram for int-valued data.");

    register_histogram_by_type<axes::regular_noflow, dense_int_storage>(hist,
        "regular_int_noflow",
        "N-dimensional histogram for int-valued data.");

    // Completely general histograms

    register_histogram_by_type<axes::any, dense_int_storage>(hist,
        "any_int",
        "N-dimensional histogram for int-valued data with any axis types.");

    register_histogram_by_type<axes::any, dense_double_storage>(hist,
        "any_double",
        "N-dimensional histogram for real-valued data with weights with any axis types.");

    register_histogram_by_type<axes::any, bh::unlimited_storage<>>(hist,
        "any_unlimited",
        "N-dimensional histogram for unlimited size data with any axis types.");

    register_histogram_by_type<axes::any, bh::weight_storage>(hist,
        "any_weight",
        "N-dimensional histogram for weighted data with any axis types.");

    m.def("make_histogram", [](py::args args, py::kwargs kwargs) -> py::object {
        axes::any values = py::cast<axes::any>(args);
        auto storage_union = extract_storage(kwargs);

        return bh::axis::visit([&values](auto&& storage) -> py::object {
            return py::cast(bh::make_histogram_with(storage, values));
        },
        storage_union);

    }, "Make any histogram");

    // register_histogram_by_type<axes::any, bh::profile_storage>(hist,
    //    "any_profile",
    //    "N-dimensional histogram for sampled data with any axis types.");

    // register_histogram_by_type<axes::any, bh::weighted_profile_storage>(hist,
    //    "any_weighted_profile",
    //    "N-dimensional histogram for weighted and sampled data with any axis types.");

}

