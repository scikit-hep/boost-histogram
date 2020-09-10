// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#include <bh_python/pybind11.hpp>

#include <bh_python/axis.hpp>
#include <bh_python/kwargs.hpp>
#include <bh_python/register_axis.hpp>
#include <bh_python/regular_numpy.hpp>
#include <boost/mp11.hpp>
#include <vector>

template <class... Ts, class Func>
void register_axis_each(py::module& mod, Func&& function) {
    using namespace boost::mp11;
    using types = mp_list<Ts...>;

    mp_for_each<mp_iota_c<sizeof...(Ts)>>([&](auto I) {
        using T = mp_at_c<types, I>;
        auto ax = register_axis<T>(mod);
        function(ax);
    });
}

void validate_axis_options(bool underflow, bool overflow, bool growth) {
    if(growth && (!underflow || !overflow)) {
        throw py::value_error(
            "growth=True with underflow=False or overflow=False is not supported");
    }
}

void register_axes(py::module& mod) {
    py::class_<options>(mod, "options")
        .def(py::init<bool, bool, bool, bool>(),
             "underflow"_a = false,
             "overflow"_a  = false,
             "circular"_a  = false,
             "growth"_a    = false)
        .def("__eq__",
             [](const options& self, const py::object& other) {
                 try {
                     return self == py::cast<options>(other);
                 } catch(const py::cast_error&) {
                     return false;
                 }
             })
        .def("__ne__",
             [](const options& self, const py::object& other) {
                 try {
                     return self != py::cast<options>(other);
                 } catch(const py::cast_error&) {
                     return true;
                 }
             })
        .def(py::pickle([](const options& op) { return py::make_tuple(op.option); },
                        [](py::tuple t) {
                            if(t.size() != 1)
                                throw std::runtime_error("Invalid state");
                            return options{py::cast<unsigned>(t[0])};
                        }))

        .def("__copy__", [](const options& self) { return options(self); })
        .def("__deepcopy__",
             [](const options& self, py::object) { return options{self}; })

        .def_property_readonly("underflow", &options::underflow)
        .def_property_readonly("overflow", &options::overflow)
        .def_property_readonly("circular", &options::circular)
        .def_property_readonly("growth", &options::growth)

        .def("__repr__", [](const options& self) {
            return py::str("options(underflow={}, overflow={}, circular={}, growth={})")
                .format(
                    self.underflow(), self.overflow(), self.circular(), self.growth());
        });

    register_axis_each<axis::regular_none,
                       axis::regular_uflow,
                       axis::regular_oflow,
                       axis::regular_uoflow,
                       axis::regular_uoflow_growth,
                       axis::regular_circular,
                       axis::regular_numpy>(mod, [](auto ax) {
        ax.def(py::init<unsigned, double, double>(), "bins"_a, "start"_a, "stop"_a);
    });

    register_axis<axis::regular_pow>(mod)
        .def(py::init([](unsigned n, double start, double stop, double pow) {
                 return new axis::regular_pow(
                     bh::axis::transform::pow{pow}, n, start, stop);
             }),
             "bins"_a,
             "start"_a,
             "stop"_a,
             "power"_a)
        .def_property_readonly("transform", [](const axis::regular_pow& self) {
            return self.transform();
        });

    register_axis<axis::regular_trans>(mod)
        .def(py::init([](unsigned n, double start, double stop, func_transform& trans) {
                 return new axis::regular_trans(trans, n, start, stop);
             }),
             "bins"_a,
             "start"_a,
             "stop"_a,
             "tranform"_a)
        .def_property_readonly("transform", [](const axis::regular_trans& self) {
            return self.transform();
        });

    register_axis_each<axis::variable_none,
                       axis::variable_uflow,
                       axis::variable_oflow,
                       axis::variable_uoflow,
                       axis::variable_uoflow_growth,
                       axis::variable_circular>(
        mod, [](auto ax) { ax.def(py::init<std::vector<double>>(), "edges"_a); });

    register_axis_each<axis::integer_none,
                       axis::integer_uflow,
                       axis::integer_oflow,
                       axis::integer_uoflow,
                       axis::integer_growth,
                       axis::integer_circular>(
        mod, [](auto ax) { ax.def(py::init<int, int>(), "start"_a, "stop"_a); });

    register_axis_each<axis::category_int, axis::category_int_growth>(
        mod, [](auto ax) { ax.def(py::init<std::vector<int>>(), "categories"_a); });

    register_axis_each<axis::category_str, axis::category_str_growth>(mod, [](auto ax) {
        ax.def(py::init<std::vector<std::string>>(), "categories"_a);
    });

    register_axis<axis::boolean>(mod, "boolean").def(py::init<>());

    ;
}
