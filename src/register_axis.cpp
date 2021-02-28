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

void register_axes(py::module& mod) {
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
