// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#include <boost/histogram/python/pybind11.hpp>

#include <boost/histogram/python/axis.hpp>
#include <boost/histogram/python/kwargs.hpp>
#include <boost/histogram/python/register_axis.hpp>
#include <boost/histogram/python/regular_numpy.hpp>
#include <boost/mp11.hpp>
#include <vector>

template <class... Ts, class Init, class... Passthrough>
void register_axis_sub_types(py::module &mod,
                             std::array<const char *, sizeof...(Ts)> names,
                             const char *doc,
                             Init &&init,
                             Passthrough &&... passthrough) {
    using namespace boost::mp11;
    using types = mp_list<Ts...>;
    mp_for_each<mp_iota_c<sizeof...(Ts)>>([&](auto I) {
        using T = mp_at_c<types, I>;
        register_axis<T>(mod, names.at(I), doc)
            .def(std::forward<Init>(init), std::forward<Passthrough>(passthrough)...);
    });
}

void validate_axis_options(bool underflow, bool overflow, bool growth) {
    if(growth && (!underflow || !overflow)) {
        throw py::value_error(
            "growth=True with underflow=False or overflow=False is not supported");
    }
}

void register_axes(py::module &mod) {
    py::class_<options>(mod, "options")
        .def(py::init<bool, bool, bool, bool>(),
             "underflow"_a = false,
             "overflow"_a  = false,
             "circular"_a  = false,
             "growth"_a    = false)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::pickle([](const options &op) { return py::make_tuple(op.option); },
                        [](py::tuple t) {
                            if(t.size() != 1)
                                throw std::runtime_error("Invalid state");
                            return options{py::cast<unsigned>(t[0])};
                        }))

        .def("__copy__", [](const options &self) { return options(self); })
        .def("__deepcopy__",
             [](const options &self, py::object) { return options{self}; })

        .def_property_readonly("underflow", &options::underflow)
        .def_property_readonly("overflow", &options::overflow)
        .def_property_readonly("circular", &options::circular)
        .def_property_readonly("growth", &options::growth)

        .def("__repr__", [](const options &self) {
            return py::str("options(underflow={}, overflow={}, circular={}, growth={})")
                .format(
                    self.underflow(), self.overflow(), self.circular(), self.growth());
        });

    register_axis_sub_types<axis::regular_none,
                            axis::regular_uflow,
                            axis::regular_oflow,
                            axis::regular_uoflow,
                            axis::regular_uoflow_growth,
                            axis::regular_numpy>(
        mod,
        {"_regular_none",
         "_regular_uflow",
         "_regular_oflow",
         "_regular_uoflow",
         "_regular_uoflow_growth",
         "_regular_numpy"},
        "Evenly spaced bins",
        py::init<unsigned, double, double, metadata_t>(),
        "bins"_a,
        "start"_a,
        "stop"_a,
        "metadata"_a = py::none());

    register_axis<axis::circular>(mod, "circular", "Evenly spaced bins with wraparound")
        .def(py::init<unsigned, double, double, metadata_t>(),
             "bins"_a,
             "start"_a,
             "stop"_a,
             "metadata"_a = py::none());

    register_axis<axis::regular_log>(mod, "regular_log", "Evenly spaced bins in log10")
        .def(py::init<unsigned, double, double, metadata_t>(),
             "bins"_a,
             "start"_a,
             "stop"_a,
             "metadata"_a = py::none());

    register_axis<axis::regular_sqrt>(mod, "regular_sqrt", "Evenly spaced bins in sqrt")
        .def(py::init<unsigned, double, double, metadata_t>(),
             "bins"_a,
             "start"_a,
             "stop"_a,
             "metadata"_a = py::none());

    register_axis<axis::regular_pow>(
        mod, "regular_pow", "Evenly spaced bins in a power")
        .def(py::init([](unsigned n,
                         double start,
                         double stop,
                         double pow,
                         metadata_t metadata) {
                 return new axis::regular_pow(
                     bh::axis::transform::pow{pow}, n, start, stop, metadata);
             }),
             "bins"_a,
             "start"_a,
             "stop"_a,
             "power"_a,
             "metadata"_a = py::none());

    register_axis_sub_types<axis::variable_none,
                            axis::variable_uflow,
                            axis::variable_oflow,
                            axis::variable_uoflow,
                            axis::variable_uoflow_growth>(
        mod,
        {"_variable_none",
         "_variable_uflow",
         "_variable_oflow",
         "_variable_uoflow",
         "_variable_uoflow_growth"},
        "Unevenly spaced bins",
        py::init<std::vector<double>, metadata_t>(),
        "edges"_a,
        "metadata"_a = py::none());

    register_axis_sub_types<axis::integer_none,
                            axis::integer_uflow,
                            axis::integer_oflow,
                            axis::integer_uoflow,
                            axis::integer_growth>(mod,
                                                  {"_integer_none",
                                                   "_integer_uflow",
                                                   "_integer_oflow",
                                                   "_integer_uoflow",
                                                   "_integer_growth"},
                                                  "Contiguous integers",
                                                  py::init<int, int, metadata_t>(),
                                                  "start"_a,
                                                  "stop"_a,
                                                  "metadata"_a = py::none());

    register_axis_sub_types<axis::category_int, axis::category_int_growth>(
        mod,
        {"_category_int", "_category_int_growth"},
        "Axis with discontiguous integer bins",
        py::init<std::vector<int>, metadata_t>(),
        "categories"_a,
        "metadata"_a = py::none());

    register_axis_sub_types<axis::category_str, axis::category_str_growth>(
        mod,
        {"_category_str", "_category_str_growth"},
        "Axis with text bins",
        py::init<std::vector<std::string>, metadata_t>(),
        "categories"_a,
        "metadata"_a = py::none());
}
