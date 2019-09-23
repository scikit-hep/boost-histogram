// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#include <boost/histogram/python/pybind11.hpp>

#include <boost/histogram/python/axis.hpp>
#include <boost/histogram/python/kwargs.hpp>
#include <boost/histogram/python/register_axis.hpp>
#include <vector>

void register_axes(py::module &ax) {
    py::class_<options>(ax, "options")
        .def_property_readonly("none", &options::none)
        .def_property_readonly("underflow", &options::underflow)
        .def_property_readonly("overflow", &options::overflow)
        .def_property_readonly("circular", &options::circular)
        .def_property_readonly("growth", &options::growth)
        .def("__repr__", [](const options &self) {
            return py::str("options(none={}, underflow={}, overflow={}, circular={}, "
                           "growth={})")
                .format(self.none(),
                        self.underflow(),
                        self.overflow(),
                        self.circular(),
                        self.growth());
        });

    register_axis<axis::_regular_uoflow>(ax, "_regular_uoflow", "Evenly spaced bins")
        .def(construct_axes<axis::_regular_uoflow, unsigned, double, double>(),
             "n"_a,
             "start"_a,
             "stop"_a,
             "metadata"_a = py::str());

    register_axis<axis::_regular_uflow>(
        ax, "_regular_uflow", "Evenly spaced bins with underflow")
        .def(construct_axes<axis::_regular_uflow, unsigned, double, double>(),
             "n"_a,
             "start"_a,
             "stop"_a,
             "metadata"_a = py::str());

    register_axis<axis::_regular_oflow>(
        ax, "_regular_oflow", "Evenly spaced bins with overflow ")
        .def(construct_axes<axis::_regular_oflow, unsigned, double, double>(),
             "n"_a,
             "start"_a,
             "stop"_a,
             "metadata"_a = py::str());

    register_axis<axis::_regular_noflow>(
        ax, "_regular_noflow", "Evenly spaced bins without over/under flow")
        .def(construct_axes<axis::_regular_noflow, unsigned, double, double>(),
             "n"_a,
             "start"_a,
             "stop"_a,
             "metadata"_a = py::str());

    register_axis<axis::_regular_growth>(
        ax, "_regular_growth", "Evenly spaced bins that grow as needed")
        .def(construct_axes<axis::_regular_growth, unsigned, double, double>(),
             "n"_a,
             "start"_a,
             "stop"_a,
             "metadata"_a = py::str());

    ax.def(
        "_make_regular",
        [](unsigned n,
           double start,
           double stop,
           metadata_t metadata,
           bool underflow,
           bool overflow,
           bool growth,
           py::kwargs kwargs) -> py::object {
            validate_metadata(metadata);

            auto flow = optional_arg(kwargs, "flow");
            finalize_args(kwargs);

            // Allow "flow" to override
            if(!flow.is_none()) {
                underflow = overflow = py::cast<bool>(flow);
            }

            if(growth) {
                return py::cast(axis::_regular_growth(n, start, stop, metadata),
                                py::return_value_policy::move);
            } else if(underflow && overflow) {
                return py::cast(axis::_regular_uoflow(n, start, stop, metadata),
                                py::return_value_policy::move);
            } else if(underflow && !overflow) {
                return py::cast(axis::_regular_uflow(n, start, stop, metadata),
                                py::return_value_policy::move);
            } else if(!underflow && overflow) {
                return py::cast(axis::_regular_oflow(n, start, stop, metadata),
                                py::return_value_policy::move);
            } else {
                return py::cast(axis::_regular_noflow(n, start, stop, metadata),
                                py::return_value_policy::move);
            }
        },
        "n"_a,
        "start"_a,
        "stop"_a,
        "metadata"_a  = py::str(),
        "underflow"_a = true,
        "overflow"_a  = true,
        "growth"_a    = false,
        "Make a regular axis with nice keyword arguments for underflow, overflow, and "
        "growth. "
        "Passing 'flow' will override underflow and overflow at the same time.");

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
        .def(py::init([](unsigned n,
                         double start,
                         double stop,
                         double pow,
                         metadata_t metadata) {
                 validate_metadata(metadata);
                 return new axis::regular_pow(
                     bh::axis::transform::pow{pow}, n, start, stop, metadata);
             }),
             "n"_a,
             "start"_a,
             "stop"_a,
             "power"_a,
             "metadata"_a = py::str());

    register_axis<axis::_variable_uoflow>(
        ax, "_variable_uoflow", "Unevenly spaced bins")
        .def(construct_axes<axis::_variable_uoflow, std::vector<double>>(),
             "edges"_a,
             "metadata"_a = py::str());

    register_axis<axis::_variable_uflow>(
        ax, "_variable_uflow", "Unevenly spaced bins with underflow")
        .def(construct_axes<axis::_variable_uflow, std::vector<double>>(),
             "edges"_a,
             "metadata"_a = py::str());

    register_axis<axis::_variable_oflow>(
        ax, "_variable_oflow", "Unevenly spaced bins with overflow")
        .def(construct_axes<axis::_variable_oflow, std::vector<double>>(),
             "edges"_a,
             "metadata"_a = py::str());

    register_axis<axis::_variable_noflow>(
        ax, "_variable_noflow", "Unevenly spaced bins without under/overflow")
        .def(construct_axes<axis::_variable_noflow, std::vector<double>>(),
             "edges"_a,
             "metadata"_a = py::str());

    ax.def(
        "_make_variable",
        [](std::vector<double> edges,
           metadata_t metadata,
           bool underflow,
           bool overflow,
           py::kwargs kwargs) -> py::object {
            validate_metadata(metadata);

            auto flow = optional_arg(kwargs, "flow");
            finalize_args(kwargs);

            // Allow "flow" to override
            if(!flow.is_none()) {
                underflow = overflow = py::cast<bool>(flow);
            }

            if(underflow && overflow) {
                return py::cast(axis::_variable_uoflow(edges, metadata),
                                py::return_value_policy::move);
            } else if(underflow && !overflow) {
                return py::cast(axis::_variable_uflow(edges, metadata),
                                py::return_value_policy::move);
            } else if(!underflow && overflow) {
                return py::cast(axis::_variable_oflow(edges, metadata),
                                py::return_value_policy::move);
            } else {
                return py::cast(axis::_variable_noflow(edges, metadata),
                                py::return_value_policy::move);
            }
        },
        "edges"_a,
        "metadata"_a  = py::str(),
        "underflow"_a = true,
        "overflow"_a  = true,
        "Make a variable binned axis with nice keyword arguments for underflow, "
        "overflow. "
        "Passing 'flow' will override underflow and overflow at the same time.");

    register_axis<axis::_integer_uoflow>(ax, "_integer_uoflow", "Contigious integers")
        .def(construct_axes<axis::_integer_uoflow, int, int>(),
             "min"_a,
             "max"_a,
             "metadata"_a = py::str());

    register_axis<axis::_integer_uflow>(
        ax, "_integer_uflow", "Contigious integers with underflow")
        .def(construct_axes<axis::_integer_uflow, int, int>(),
             "min"_a,
             "max"_a,
             "metadata"_a = py::str());

    register_axis<axis::_integer_oflow>(
        ax, "_integer_oflow", "Contigious integers with overflow")
        .def(construct_axes<axis::_integer_oflow, int, int>(),
             "min"_a,
             "max"_a,
             "metadata"_a = py::str());

    register_axis<axis::_integer_noflow>(
        ax, "_integer_noflow", "Contigious integers with no under/overflow")
        .def(construct_axes<axis::_integer_noflow, int, int>(),
             "min"_a,
             "max"_a,
             "metadata"_a = py::str());

    register_axis<axis::_integer_growth>(
        ax, "_integer_growth", "Contigious integers with growth")
        .def(construct_axes<axis::_integer_growth, int, int>(),
             "min"_a,
             "max"_a,
             "metadata"_a = py::str());

    ax.def(
        "_make_integer",
        [](int start,
           int stop,
           metadata_t metadata,
           bool underflow,
           bool overflow,
           bool growth,
           py::kwargs kwargs) -> py::object {
            validate_metadata(metadata);

            auto flow = optional_arg(kwargs, "flow");
            finalize_args(kwargs);

            // Allow "flow" to override
            if(!flow.is_none()) {
                underflow = overflow = py::cast<bool>(flow);
            }

            if(growth) {
                return py::cast(axis::_integer_growth(start, stop, metadata),
                                py::return_value_policy::move);
            } else if(underflow && overflow) {
                return py::cast(axis::_integer_uoflow(start, stop, metadata),
                                py::return_value_policy::move);
            } else if(underflow && !overflow) {
                return py::cast(axis::_integer_uflow(start, stop, metadata),
                                py::return_value_policy::move);
            } else if(!underflow && overflow) {
                return py::cast(axis::_integer_oflow(start, stop, metadata),
                                py::return_value_policy::move);
            } else {
                return py::cast(axis::_integer_noflow(start, stop, metadata),
                                py::return_value_policy::move);
            }
        },
        "start"_a,
        "stop"_a,
        "metadata"_a  = py::str(),
        "underflow"_a = true,
        "overflow"_a  = true,
        "growth"_a    = false,
        "Make an integer axis with nice keyword arguments for underflow, overflow, and "
        "growth. "
        "Passing 'flow' will override underflow and overflow at the same time.");

    register_axis<axis::_category_int>(ax, "_category_int", "Text label bins")
        .def(construct_axes<axis::_category_int, std::vector<int>>(),
             "labels"_a,
             "metadata"_a = py::str());

    register_axis<axis::_category_int_growth>(
        ax, "_category_int_growth", "Text label bins")
        .def(construct_axes<axis::_category_int_growth, std::vector<int>>(),
             "labels"_a,
             "metadata"_a = py::str())
        .def(py::init<>());

    register_axis<axis::_category_str>(ax, "_category_str", "Text label bins")
        .def(construct_axes<axis::_category_str, std::vector<std::string>>(),
             "labels"_a,
             "metadata"_a = py::str());

    register_axis<axis::_category_str_growth>(
        ax, "_category_str_growth", "Text label bins")
        .def(construct_axes<axis::_category_str_growth, std::vector<std::string>>(),
             "labels"_a,
             "metadata"_a = py::str())
        .def(py::init<>());

    ax.def(
        "_make_category",
        [](py::object labels, metadata_t metadata, bool growth) -> py::object {
            validate_metadata(metadata);

            try {
                auto int_values = py::cast<std::vector<int>>(labels);

                if(growth) {
                    return py::cast(axis::_category_int_growth(int_values, metadata),
                                    py::return_value_policy::move);
                } else {
                    return py::cast(axis::_category_int(int_values, metadata),
                                    py::return_value_policy::move);
                }
            } catch(const py::cast_error &) {
                auto str_values = py::cast<std::vector<std::string>>(labels);

                if(growth) {
                    return py::cast(axis::_category_str_growth(str_values, metadata),
                                    py::return_value_policy::move);
                } else {
                    return py::cast(axis::_category_str(str_values, metadata),
                                    py::return_value_policy::move);
                }
            }
        },
        "labels"_a,
        "metadata"_a = py::str(),
        "growth"_a   = false,
        "Make an category axis with a nice keyword argument for growth. Int and string "
        "supported.");
}
