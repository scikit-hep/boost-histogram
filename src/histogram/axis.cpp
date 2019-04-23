// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#include <boost/histogram/python/pybind11.hpp>

#include <boost/histogram/python/axis.hpp>
#include <boost/histogram/python/kwargs.hpp>
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
        .def(construct_axes<axis::regular_uoflow, unsigned, double, double>(),
             "n"_a,
             "start"_a,
             "stop"_a,
             "metadata"_a = py::str());

    register_axis<axis::regular_uflow>(ax, "regular_uflow", "Evenly spaced bins with underflow")
        .def(construct_axes<axis::regular_uflow, unsigned, double, double>(),
             "n"_a,
             "start"_a,
             "stop"_a,
             "metadata"_a = py::str());

    register_axis<axis::regular_oflow>(ax, "regular_oflow", "Evenly spaced bins with overflow ")
        .def(construct_axes<axis::regular_oflow, unsigned, double, double>(),
             "n"_a,
             "start"_a,
             "stop"_a,
             "metadata"_a = py::str());

    register_axis<axis::regular_noflow>(ax, "regular_noflow", "Evenly spaced bins without over/under flow")
        .def(construct_axes<axis::regular_noflow, unsigned, double, double>(),
             "n"_a,
             "start"_a,
             "stop"_a,
             "metadata"_a = py::str());

    register_axis<axis::regular_growth>(ax, "regular_growth", "Evenly spaced bins that grow as needed")
        .def(construct_axes<axis::regular_growth, unsigned, double, double>(),
             "n"_a,
             "start"_a,
             "stop"_a,
             "metadata"_a = py::str());

    ax.def(
        "make_regular",
        [](unsigned n,
           double start,
           double stop,
           metadata_t metadata,
           bool underflow,
           bool overflow,
           bool growth,
           py::kwargs kwargs) -> py::object {
            validate_metadata(metadata);

            std::unique_ptr<bool> flow = optional_arg<bool>(kwargs, "flow");
            finalize_args(kwargs);

            // Allow "flow" to override
            if(flow) {
                underflow = *flow;
                overflow  = *flow;
            }

            if(growth) {
                return py::cast(axis::regular_growth(n, start, stop, metadata), py::return_value_policy::move);
            } else if(underflow && overflow) {
                return py::cast(axis::regular_uoflow(n, start, stop, metadata), py::return_value_policy::move);
            } else if(underflow && !overflow) {
                return py::cast(axis::regular_uflow(n, start, stop, metadata), py::return_value_policy::move);
            } else if(!underflow && overflow) {
                return py::cast(axis::regular_oflow(n, start, stop, metadata), py::return_value_policy::move);
            } else {
                return py::cast(axis::regular_noflow(n, start, stop, metadata), py::return_value_policy::move);
            }
        },
        "n"_a,
        "start"_a,
        "stop"_a,
        "metadata"_a  = py::str(),
        "underflow"_a = true,
        "overflow"_a  = true,
        "growth"_a    = false,
        "Make a regular axis with nice keyword arguments for underflow, overflow, and growth. "
        "Passing 'flow' will override underflow and overflow at the same time.");

    ax.attr("regular") = factory_meta_py(ax.attr("make_regular"),
                                         py::make_tuple(ax.attr("regular_uoflow"),
                                                        ax.attr("regular_uflow"),
                                                        ax.attr("regular_oflow"),
                                                        ax.attr("regular_noflow"),
                                                        ax.attr("regular_growth")));

    register_axis<axis::circular>(ax, "circular", "Evenly spaced bins with wraparound")
        .def(construct_axes<axis::circular, unsigned, double, double>(),
             "n"_a,
             "start"_a,
             "stop"_a,
             "metadata"_a = py::str())
        .def(py::init([](unsigned n, double stop, metadata_t metadata) {
                 validate_metadata(metadata);
                 return new axis::circular{n, 0.0, stop, metadata};
             }),
             "n"_a,
             "stop"_a,
             "metadata"_a = py::str());

    register_axis<axis::regular_log>(ax, "regular_log", "Evenly spaced bins in log10")
        .def(construct_axes<axis::regular_log, unsigned, double, double>(),
             "n"_a,
             "start"_a,
             "stop"_a,
             "metadata"_a = py::str());

    register_axis<axis::regular_sqrt>(ax, "regular_sqrt", "Evenly spaced bins in sqrt")
        .def(construct_axes<axis::regular_sqrt, unsigned, double, double>(),
             "n"_a,
             "start"_a,
             "stop"_a,
             "metadata"_a = py::str());

    register_axis<axis::regular_pow>(ax, "regular_pow", "Evenly spaced bins in a power")
        .def(py::init([](unsigned n, double start, double stop, double pow, metadata_t metadata) {
                 validate_metadata(metadata);
                 return new axis::regular_pow(bh::axis::transform::pow{pow}, n, start, stop, metadata);
             }),
             "n"_a,
             "start"_a,
             "stop"_a,
             "power"_a,
             "metadata"_a = py::str());

    register_axis<axis::variable_uoflow>(ax, "variable_uoflow", "Unevenly spaced bins")
        .def(construct_axes<axis::variable_uoflow, std::vector<double>>(), "edges"_a, "metadata"_a = py::str());

    register_axis<axis::variable_uflow>(ax, "variable_uflow", "Unevenly spaced bins with underflow")
        .def(construct_axes<axis::variable_uflow, std::vector<double>>(), "edges"_a, "metadata"_a = py::str());

    register_axis<axis::variable_oflow>(ax, "variable_oflow", "Unevenly spaced bins with overflow")
        .def(construct_axes<axis::variable_oflow, std::vector<double>>(), "edges"_a, "metadata"_a = py::str());

    register_axis<axis::variable_noflow>(ax, "variable_noflow", "Unevenly spaced bins without under/overflow")
        .def(construct_axes<axis::variable_noflow, std::vector<double>>(), "edges"_a, "metadata"_a = py::str());

    ax.def(
        "make_variable",
        [](std::vector<double> edges, metadata_t metadata, bool underflow, bool overflow, py::kwargs kwargs)
            -> py::object {
            validate_metadata(metadata);

            std::unique_ptr<bool> flow = optional_arg<bool>(kwargs, "flow");
            finalize_args(kwargs);

            // Allow "flow" to override
            if(flow) {
                underflow = *flow;
                overflow  = *flow;
            }

            if(underflow && overflow) {
                return py::cast(axis::variable_uoflow(edges, metadata), py::return_value_policy::move);
            } else if(underflow && !overflow) {
                return py::cast(axis::variable_uflow(edges, metadata), py::return_value_policy::move);
            } else if(!underflow && overflow) {
                return py::cast(axis::variable_oflow(edges, metadata), py::return_value_policy::move);
            } else {
                return py::cast(axis::variable_noflow(edges, metadata), py::return_value_policy::move);
            }
        },
        "edges"_a,
        "metadata"_a  = py::str(),
        "underflow"_a = true,
        "overflow"_a  = true,
        "Make a variable binned axis with nice keyword arguments for underflow, overflow. "
        "Passing 'flow' will override underflow and overflow at the same time.");

    ax.attr("variable") = factory_meta_py(ax.attr("make_variable"),
                                          py::make_tuple(ax.attr("variable_uoflow"),
                                                         ax.attr("variable_uflow"),
                                                         ax.attr("variable_oflow"),
                                                         ax.attr("variable_noflow")));

    register_axis<axis::integer_uoflow>(ax, "integer_uoflow", "Contigious integers")
        .def(construct_axes<axis::integer_uoflow, int, int>(), "min"_a, "max"_a, "metadata"_a = py::str());

    register_axis<axis::integer_uflow>(ax, "integer_uflow", "Contigious integers with underflow")
        .def(construct_axes<axis::integer_uflow, int, int>(), "min"_a, "max"_a, "metadata"_a = py::str());

    register_axis<axis::integer_oflow>(ax, "integer_oflow", "Contigious integers with overflow")
        .def(construct_axes<axis::integer_oflow, int, int>(), "min"_a, "max"_a, "metadata"_a = py::str());

    register_axis<axis::integer_noflow>(ax, "integer_noflow", "Contigious integers with no under/overflow")
        .def(construct_axes<axis::integer_noflow, int, int>(), "min"_a, "max"_a, "metadata"_a = py::str());

    register_axis<axis::integer_growth>(ax, "integer_growth", "Contigious integers with growth")
        .def(construct_axes<axis::integer_growth, int, int>(), "min"_a, "max"_a, "metadata"_a = py::str());

    ax.def(
        "make_integer",
        [](int start, int stop, metadata_t metadata, bool underflow, bool overflow, bool growth, py::kwargs kwargs)
            -> py::object {
            validate_metadata(metadata);

            std::unique_ptr<bool> flow = optional_arg<bool>(kwargs, "flow");
            finalize_args(kwargs);

            // Allow "flow" to override
            if(flow) {
                underflow = *flow;
                overflow  = *flow;
            }

            if(growth) {
                return py::cast(axis::integer_growth(start, stop, metadata), py::return_value_policy::move);
            } else if(underflow && overflow) {
                return py::cast(axis::integer_uoflow(start, stop, metadata), py::return_value_policy::move);
            } else if(underflow && !overflow) {
                return py::cast(axis::integer_uflow(start, stop, metadata), py::return_value_policy::move);
            } else if(!underflow && overflow) {
                return py::cast(axis::integer_oflow(start, stop, metadata), py::return_value_policy::move);
            } else {
                return py::cast(axis::integer_noflow(start, stop, metadata), py::return_value_policy::move);
            }
        },
        "start"_a,
        "stop"_a,
        "metadata"_a  = py::str(),
        "underflow"_a = true,
        "overflow"_a  = true,
        "growth"_a    = false,
        "Make an integer axis with nice keyword arguments for underflow, overflow, and growth. "
        "Passing 'flow' will override underflow and overflow at the same time.");

    ax.attr("integer") = factory_meta_py(ax.attr("make_integer"),
                                         py::make_tuple(ax.attr("integer_uoflow"),
                                                        ax.attr("integer_uflow"),
                                                        ax.attr("integer_oflow"),
                                                        ax.attr("integer_noflow"),
                                                        ax.attr("integer_growth")));

    register_axis<axis::category_int>(ax, "category_int", "Text label bins")
        .def(construct_axes<axis::category_int, std::vector<int>>(), "labels"_a, "metadata"_a = py::str());

    register_axis<axis::category_int_growth>(ax, "category_int_growth", "Text label bins")
        .def(construct_axes<axis::category_int_growth, std::vector<int>>(), "labels"_a, "metadata"_a = py::str())
        .def(py::init<>());

    register_axis<axis::category_str>(ax, "category_str", "Text label bins")
        .def(construct_axes<axis::category_str, std::vector<std::string>>(), "labels"_a, "metadata"_a = py::str());

    register_axis<axis::category_str_growth>(ax, "category_str_growth", "Text label bins")
        .def(
            construct_axes<axis::category_str_growth, std::vector<std::string>>(), "labels"_a, "metadata"_a = py::str())
        .def(py::init<>());

    ax.def(
        "make_category",
        [](py::object labels, metadata_t metadata, bool growth) -> py::object {
            validate_metadata(metadata);

            try {
                auto int_values = py::cast<std::vector<int>>(labels);

                if(growth) {
                    return py::cast(axis::category_int_growth(int_values, metadata), py::return_value_policy::move);
                } else {
                    return py::cast(axis::category_int(int_values, metadata), py::return_value_policy::move);
                }
            } catch(const py::cast_error &) {
                auto str_values = py::cast<std::vector<std::string>>(labels);

                if(growth) {
                    return py::cast(axis::category_str_growth(str_values, metadata), py::return_value_policy::move);
                } else {
                    return py::cast(axis::category_str(str_values, metadata), py::return_value_policy::move);
                }
            }
        },
        "labels"_a,
        "metadata"_a = py::str(),
        "growth"_a   = false,
        "Make an category axis with a nice keyword argument for growth. Int and string supported.");

    ax.attr("category") = factory_meta_py(ax.attr("make_category"),
                                          py::make_tuple(ax.attr("category_int"),
                                                         ax.attr("category_int_growth"),
                                                         ax.attr("category_str"),
                                                         ax.attr("category_str_growth")));

    return ax;
}
