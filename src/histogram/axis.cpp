// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#include <boost/histogram/python/pybind11.hpp>

#include <boost/histogram/python/axis.hpp>
#include <boost/histogram/python/register_axis.hpp>

#include <vector>

py::module register_axes(py::module &m) {
    py::module ax = m.def_submodule("axis");

    py::module opt = ax.def_submodule("options");

    opt.attr("none")      = (unsigned)bh::axis::option::none;
    opt.attr("underflow") = (unsigned)bh::axis::option::underflow;
    opt.attr("overflow")  = (unsigned)bh::axis::option::overflow;
    opt.attr("circular")  = (unsigned)bh::axis::option::circular;
    opt.attr("growth")    = (unsigned)bh::axis::option::growth;

    // This factory makes a class that can be used to create axes and also be used in is_instance
    py::object factory_meta_py = py::module::import("boost.histogram_utils").attr("FactoryMeta");

    register_axis<axis::regular_uoflow>(ax, "regular_uoflow", "Evenly spaced bins")
        .def(py::init<unsigned, double, double, metadata_t>(), "n"_a, "start"_a, "stop"_a, "metadata"_a = py::str());
    
    register_axis<axis::regular_noflow>(ax, "regular_noflow", "Evenly spaced bins without over/under flow")
        .def(py::init<unsigned, double, double, metadata_t>(), "n"_a, "start"_a, "stop"_a, "metadata"_a = py::str());
    
    register_axis<axis::regular_growth>(ax, "regular_growth", "Evenly spaced bins that grow as needed")
        .def(py::init<unsigned, double, double, metadata_t>(), "n"_a, "start"_a, "stop"_a, "metadata"_a = py::str());
    
    ax.def(
        "make_regular",
        [](unsigned n, double start, double stop, metadata_t metadata, bool flow, bool growth) -> py::object {
            if(growth) {
                return py::cast(axis::regular_growth(n, start, stop, metadata), py::return_value_policy::move);
            } else if(flow) {
                return py::cast(axis::regular_uoflow(n, start, stop, metadata), py::return_value_policy::move);
            } else {
                return py::cast(axis::regular_noflow(n, start, stop, metadata), py::return_value_policy::move);
            }
        },
        "n"_a,
        "start"_a,
        "stop"_a,
        "metadata"_a = py::str(),
        "flow"_a     = true,
        "growth"_a   = false,
        "Make a regular axis with nice keyword arguments for flow and growth");

    ax.attr("regular") = factory_meta_py(
        ax.attr("make_regular"),
        py::make_tuple(ax.attr("regular_uoflow"), ax.attr("regular_noflow"), ax.attr("regular_growth")));

    register_axis<axis::circular>(ax, "circular", "Evenly spaced bins with wraparound")
        .def(py::init<unsigned, double, double, metadata_t>(), "n"_a, "start"_a, "stop"_a, "metadata"_a = py::str())
        .def(py::init([](unsigned n, double stop, metadata_t metadata) {
                 return new axis::circular{n, 0.0, stop, metadata};
             }),
             "n"_a,
             "stop"_a,
             "metadata"_a = py::str());
    
    register_axis<axis::regular_log>(ax, "regular_log", "Evenly spaced bins in log10")
        .def(py::init<unsigned, double, double, metadata_t>(), "n"_a, "start"_a, "stop"_a, "metadata"_a = py::str());
    
    register_axis<axis::regular_sqrt>(ax, "regular_sqrt", "Evenly spaced bins in sqrt")
        .def(py::init<unsigned, double, double, metadata_t>(), "n"_a, "start"_a, "stop"_a, "metadata"_a = py::str());
    
    register_axis<axis::regular_pow>(ax, "regular_pow", "Evenly spaced bins in a power")
        .def(py::init([](unsigned n, double start, double stop, double pow, metadata_t metadata) {
                 return new axis::regular_pow(bh::axis::transform::pow{pow}, n, start, stop, metadata);
             }),
             "n"_a,
             "start"_a,
             "stop"_a,
             "power"_a,
             "metadata"_a = py::str());
    
    register_axis<axis::variable>(ax, "variable", "Unevenly spaced bins")
        .def(py::init<std::vector<double>, metadata_t>(), "edges"_a, "metadata"_a = py::str());
    
    register_axis<axis::integer_uoflow>(ax, "integer_uoflow", "Contigious integers")
        .def(py::init<int, int, metadata_t>(), "min"_a, "max"_a, "metadata"_a = py::str());
    
    register_axis<axis::integer_noflow>(ax, "integer_noflow", "Contigious integers with no under/overflow")
        .def(py::init<int, int, metadata_t>(), "min"_a, "max"_a, "metadata"_a = py::str());
    
    register_axis<axis::integer_growth>(ax, "integer_growth", "Contigious integers with growth")
        .def(py::init<int, int, metadata_t>(), "min"_a, "max"_a, "metadata"_a = py::str());
    
    register_axis<axis::category_int>(ax, "category_int", "Text label bins")
        .def(py::init<std::vector<int>, metadata_t>(), "labels"_a, "metadata"_a = py::str());

    register_axis<axis::category_int_growth>(ax, "category_int_growth", "Text label bins")
        .def(py::init<std::vector<int>, metadata_t>(), "labels"_a, "metadata"_a = py::str())
        .def(py::init<>());

    register_axis<axis::category_str>(ax, "category_str", "Text label bins")
        .def(py::init<std::vector<std::string>, metadata_t>(), "labels"_a, "metadata"_a = py::str());

    register_axis<axis::category_str_growth>(ax, "category_str_growth", "Text label bins")
        .def(py::init<std::vector<std::string>, metadata_t>(), "labels"_a, "metadata"_a = py::str())
        // Add way to allow empty list of strings
        .def(py::init<>());

    return ax;
}
