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
#include <boost/histogram/python/histogram_atomic.hpp>
#include <boost/histogram/python/histogram_threaded.hpp>

#include <boost/histogram.hpp>
#include <boost/histogram/axis/ostream.hpp>
#include <boost/histogram/algorithm/sum.hpp>
#include <boost/histogram/ostream.hpp>
#include <boost/histogram/algorithm/project.hpp>

#include <boost/mp11.hpp>

#include <cassert>
#include <vector>
#include <sstream>
#include <tuple>
#include <cmath>


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
    ;

    // Atomics for example do not support these operations
    def_optionally(hist, bh::detail::has_operator_rmul<histogram_t, double>{}, py::self *= double());
    def_optionally(hist, bh::detail::has_operator_rmul<histogram_t, double>{}, py::self * double());
    def_optionally(hist, bh::detail::has_operator_rmul<histogram_t, double>{}, double() * py::self);
    def_optionally(hist, bh::detail::has_operator_rdiv<histogram_t, double>{}, py::self /= double());
    def_optionally(hist, bh::detail::has_operator_rdiv<histogram_t, double>{}, py::self / double());

    hist.def("to_numpy", [](histogram_t& h, bool flow){
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

    .def("view", [](histogram_t& h, bool flow){
        return py::array(make_buffer(h, flow));
    }, "flow"_a = false,
         "Return a view into the data, optionally with overflow turned on")
    
    .def("axis",
        [](histogram_t &self, int i){
            unsigned ii = i < 0 ? self.rank() - (unsigned) std::abs(i) : (unsigned) i;
            if(ii < self.rank())
                return self.axis(ii);
            else
                throw std::out_of_range("The axis value must be less than the rank");
        },
     "Get N-th axis with runtime index", "i"_a,
         py::return_value_policy::move)

    // generic fill for 1 to N args
    .def("__call__",
      [](histogram_t &self, py::args args) {
        boost::mp11::mp_with_index<BOOST_HISTOGRAM_DETAIL_AXES_LIMIT>(args.size(), fill_helper<histogram_t>(self, args));
      }, "Insert data into histogram")
    
    // generic threaded fill for 1 to N args
    .def("threaded_fill",
         [](histogram_t &self, unsigned threads, py::args args) {
             boost::mp11::mp_with_index<BOOST_HISTOGRAM_DETAIL_AXES_LIMIT>(args.size(), fill_helper_threaded<histogram_t>(self, args, threads));
         }, "threads"_a, "Insert data into histogram in threads (0 for machine cores)")

    .def("at", [](histogram_t &self, py::args &args) {
        // Optimize for no dynamic?
        auto int_args = py::cast<std::vector<int>>(args);
        return self.at(int_args);
        },
         "Access bin counter at indices")

    .def("__repr__", shift_to_string<histogram_t>())
    
    .def("sum", [](const histogram_t &self) {
        return bh::algorithm::sum(self);
    })
    
    /* Broken: Does not work if any string axes present (even just in variant)
    .def("rebin", [](const histogram_t &self, unsigned axis, unsigned merge){
        return bh::algorithm::reduce(self, bh::algorithm::rebin(axis, merge));
    }, "axis"_a, "merge"_a, "Rebin by merging bins. You must select an axis.")
    
    .def("shrink", [](const histogram_t &self, unsigned axis, double lower, double upper){
        return bh::algorithm::reduce(self, bh::algorithm::shrink(axis, lower, upper));
    }, "axis"_a, "lower"_a, "upper"_a, "Shrink an axis. You must select an axis.")
    
    .def("shrink_and_rebin", [](const histogram_t &self, unsigned axis, double lower, double upper, unsigned merge){
        return bh::algorithm::reduce(self, bh::algorithm::shrink_and_rebin(axis, lower, upper, merge));
    }, "axis"_a, "lower"_a, "upper"_a, "merge"_a, "Shrink an axis and rebin. You must select an axis.")
    */
    
    .def("project", [](const histogram_t &self, py::args values){
        return bh::algorithm::project(self, py::cast<std::vector<unsigned>>(values));
    }, "Project to a single axis or several axes on a multidiminsional histogram")
    
    ;

    return hist;
}

template<typename A, typename S>
void add_atomic_fill(py::class_<bh::histogram<A, S>>& hist) {
    using histogram_t = bh::histogram<A, S>;
    hist.def("atomic_fill", [](histogram_t &self, size_t threads, py::args args) {
        boost::mp11::mp_with_index<BOOST_HISTOGRAM_DETAIL_AXES_LIMIT>(args.size(), fill_helper_atomic<histogram_t>(self, args, threads));
    }, "threads"_a, "Insert data into histogram, in some number of threads (0 for default)")
    ;
}

void register_histogram(py::module& m) {
    
    m.attr("BOOST_HISTOGRAM_DETAIL_AXES_LIMIT") = BOOST_HISTOGRAM_DETAIL_AXES_LIMIT;
    
    py::module hist = m.def_submodule("hist");


    // Fast specializations - uniform types

    register_histogram_by_type<axes::regular, storage::unlimited>(hist,
        "regular_unlimited",
        "N-dimensional histogram for real-valued data.");

    register_histogram_by_type<axes::regular, storage::int_>(hist,
        "regular_int",
        "N-dimensional histogram for int-valued data.");
    
    auto regular_atomic_int = register_histogram_by_type<axes::regular, storage::atomic_int>(hist,
        "regular_atomic_int",
        "N-dimensional histogram for atomic int-valued data.");
    add_atomic_fill(regular_atomic_int);

    register_histogram_by_type<axes::regular_noflow, storage::int_>(hist,
        "regular_noflow_int",
        "N-dimensional histogram for int-valued data.");

    // Completely general histograms

    register_histogram_by_type<axes::any, storage::int_>(hist,
        "any_int",
        "N-dimensional histogram for int-valued data with any axis types.");
    
    auto any_atomic_int = register_histogram_by_type<axes::any, storage::atomic_int>(hist,
        "any_atomic_int",
        "N-dimensional histogram for int-valued data with any axis types (threadsafe).");
    add_atomic_fill(any_atomic_int);

    register_histogram_by_type<axes::any, storage::double_>(hist,
        "any_double",
        "N-dimensional histogram for real-valued data with weights with any axis types.");

    register_histogram_by_type<axes::any, storage::unlimited>(hist,
        "any_unlimited",
        "N-dimensional histogram for unlimited size data with any axis types.");

    register_histogram_by_type<axes::any, storage::weight>(hist,
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

