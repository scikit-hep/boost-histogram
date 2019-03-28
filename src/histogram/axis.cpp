// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#include <boost/histogram/python/pybind11.hpp>

#include <pybind11/numpy.h>
#include <pybind11/operators.h>

#include <boost/histogram/python/axis.hpp>

#include <boost/histogram/axis/ostream.hpp>
#include <boost/histogram.hpp>

#include <boost/histogram/axis/traits.hpp>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

using namespace std::literals;

namespace bh = boost::histogram;

/// Add items to an axis where the axis values are continious
template<typename A, typename B>
void add_to_axis(B&& axis, std::false_type) {
    axis.def("bin", &A::bin, "The bin details (center, lower, upper)", "idx"_a, py::keep_alive<0, 1>());
    axis.def("index", py::vectorize(&A::index), "The index at a point(s) on the axis", "x"_a);
    axis.def("value", py::vectorize(&A::value), "The value(s) for a fractional bin(s) in the axis", "i"_a);

    axis.def("edges", [](const A& ax, bool flow){
        return axis_to_edges(ax, flow);
    }, "flow"_a = false, "The bin edges (length: bins + 1) (include over/underflow if flow=True)");

    axis.def("centers", [](const A& ax){
        py::array_t<double> centers((unsigned) ax.size());
        //std::vector<double> centers;
        //centers.reserve((unsigned) ax.size());
        //for(const auto& val : ax) {
            //centers.push_back(val.center());
        //}
        std::transform(ax.begin(), ax.end(), centers.mutable_data(), [](const auto& bin){return bin.center();});
        return centers;
    }, "Return the bin centers");
}

/// Add items to an axis where the axis values are not continious (categories of strings, for example)
template<typename A, typename B>
void add_to_axis(B&& axis, std::true_type) {
    axis.def("bin", &A::bin, "The bin name", "idx"_a);
    // Not that these really just don't work with string labels; they would work for numerical labels.
    axis.def("index", &A::index, "The index at a point on the axis", "x"_a);
    axis.def("value", &A::value, "The value for a fractional bin in the axis", "i"_a);
}

/// Add helpers common to all axis types
template<typename A, typename R=int>
py::class_<A> register_axis_by_type(py::module& m, const char* name, const char* desc) {
    py::class_<A> axis(m, name, desc);

    // using value_type = decltype(A::value(1.0));

    axis
    .def("__repr__", [](A &self){
        std::ostringstream out;
        out << self;
        return out.str();
    })

    .def(py::self == py::self)
    .def(py::self != py::self)

    .def("size", &A::size, "Returns the number of bins, without over- or underflow")
    .def("extent", [](const A& self){return bh::axis::traits::extent(self);},
         "Retuns the number of bins, including over- or underflow")
    .def("update", &A::update, "Bin and add a value if allowed", "i"_a)
    .def_static("options", &A::options, "Return the options associated to the axis")
    .def_property("metadata",
                  [](const A& self){return self.metadata();},
                  [](A& self, metadata_t label){self.metadata() = label;},
                  "Set the axis label")

    ;

    // We only need keepalive if this is a reference.
    using Result = decltype(std::declval<A>().bin(std::declval<int>()));

    // This is a replacement for constexpr if
    add_to_axis<A>(axis, std::integral_constant<bool, std::is_reference<Result>::value || std::is_integral<Result>::value>{});

    return axis;
}

/// Add helpers common to all types with a range of values
template<typename A, typename R=int>
py::class_<bh::axis::interval_view<A>> register_axis_iv_by_type(py::module& m, const char* name) {
    using A_iv = bh::axis::interval_view<A>;
    py::class_<A_iv> axis_iv = py::class_<A_iv>(m, name, "Lightweight bin view");

    axis_iv
    .def("upper", &A_iv::upper)
    .def("lower", &A_iv::lower)
    .def("center", &A_iv::center)
    .def("width", &A_iv::width)
    .def(py::self == py::self)
    .def(py::self != py::self)
    .def("__repr__", [](const A_iv& self){
        return "<bin ["s + std::to_string(self.lower()) + ", "s + std::to_string(self.upper()) + "]>"s;
    })
    ;

    return axis_iv;
}


void register_axis(py::module &m) {

    py::module ax = m.def_submodule("axis");

    py::module opt = ax.def_submodule("options");

    opt.attr("none") =      (unsigned) bh::axis::option::none;
    opt.attr("underflow") = (unsigned) bh::axis::option::underflow;
    opt.attr("overflow") =  (unsigned) bh::axis::option::overflow;
    opt.attr("circular") =  (unsigned) bh::axis::option::circular;
    opt.attr("growth") =    (unsigned) bh::axis::option::growth;
    
    
    register_axis_by_type<axis::regular>(ax, "regular", "Evenly spaced bins")
    .def(py::init<unsigned, double, double, metadata_t>(), "n"_a, "start"_a, "stop"_a, "metadata"_a = "")
    ;
    register_axis_iv_by_type<axis::regular>(ax, "_regular_internal_view");


    register_axis_by_type<axis::regular_noflow>(ax, "regular_noflow", "Evenly spaced bins without over/under flow")
    .def(py::init<unsigned, double, double, metadata_t>(), "n"_a, "start"_a, "stop"_a, "metadata"_a = "")
    ;
    register_axis_iv_by_type<axis::regular_noflow>(ax, "_regular_noflow_internal_view");


    register_axis_by_type<axis::regular_growth>(ax, "regular_growth", "Evenly spaced bins that grow as needed")
    .def(py::init<unsigned, double, double, metadata_t>(), "n"_a, "start"_a, "stop"_a, "metadata"_a = "")
    ;
    register_axis_iv_by_type<axis::regular_growth>(ax, "_regular_growth_internal_view");


    register_axis_by_type<axis::circular>(ax, "circular", "Evenly spaced bins with wraparound")
    .def(py::init<unsigned, double, double, metadata_t>(), "n"_a, "start"_a, "stop"_a, "metadata"_a = "")
    ;
    register_axis_iv_by_type<axis::circular>(ax, "_circular_internal_view");


    register_axis_by_type<axis::regular_log>(ax, "regular_log", "Evenly spaced bins in log10")
    .def(py::init<unsigned, double, double, metadata_t>(), "n"_a, "start"_a, "stop"_a, "metadata"_a = "")
    ;
    register_axis_iv_by_type<axis::regular_log>(ax, "_regular_log_internal_view");


    register_axis_by_type<axis::regular_sqrt>(ax, "regular_sqrt", "Evenly spaced bins in sqrt")
    .def(py::init<unsigned, double, double, metadata_t>(), "n"_a, "start"_a, "stop"_a, "metadata"_a = "")
    ;
    register_axis_iv_by_type<axis::regular_sqrt>(ax, "_regular_sqrt_internal_view");


    register_axis_by_type<axis::regular_pow>(ax, "regular_pow", "Evenly spaced bins in a power")
    .def(py::init([](double pow, unsigned n, double start, double stop, metadata_t metadata){
        return new axis::regular_pow(bh::axis::transform::pow{pow}, n, start, stop, metadata);} ),
         "pow"_a, "n"_a, "start"_a, "stop"_a, "metadata"_a = "")
    ;
    register_axis_iv_by_type<axis::regular_pow>(ax, "_regular_pow_internal_view");


    register_axis_by_type<axis::variable>(ax, "variable", "Unevenly spaced bins")
    .def(py::init<std::vector<double>, metadata_t>(), "edges"_a, "metadata"_a = "")
    ;
    register_axis_iv_by_type<axis::variable>(ax, "_variable_internal_view");


    register_axis_by_type<axis::integer>(ax, "integer", "Contigious integers")
    .def(py::init<int, int, metadata_t>(), "min"_a, "max"_a, "metadata"_a = "")
    ;
    register_axis_iv_by_type<axis::integer>(ax, "_integer_internal_view");


    register_axis_by_type<axis::category_str, std::string>(ax, "category_str", "Text label bins")
    .def(py::init<std::vector<std::string>, metadata_t>(), "labels"_a, "metadata"_a = "")
    ;

    register_axis_by_type<axis::category_str_growth, std::string>(ax, "category_str_growth", "Text label bins")
    .def(py::init<std::vector<std::string>, metadata_t>(), "labels"_a, "metadata"_a = "")
    // Add way to allow empty list of strings
    .def(py::init<>())
    ;

}
