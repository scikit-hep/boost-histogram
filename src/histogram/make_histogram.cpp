// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#include <boost/histogram/python/pybind11.hpp>
#include <pybind11/operators.h>

#include <boost/histogram/python/axis.hpp>
#include <boost/histogram/python/histogram.hpp>
#include <boost/histogram/python/kwargs.hpp>
#include <boost/histogram/python/storage.hpp>
#include <boost/histogram/python/try_cast.hpp>

#include <boost/histogram.hpp>

#include <boost/mp11.hpp>

#include <vector>

void register_make_histogram(py::module &m, py::module &hist) {
    m.def(
        "make_histogram",
        [](py::args t_args, py::kwargs kwargs) -> py::object {
            py::list args      = py::cast<py::list>(t_args);
            py::object storage = optional_arg(kwargs, "storage", py::cast(storage::int_()));
            finalize_args(kwargs);

            // Process the args as necessary for extra shortcuts
            for(size_t i = 0; i < args.size(); i++) {
                // If length three tuples are provided, make regular bins
                if(py::isinstance<py::tuple>(args[i])) {
                    py::tuple arg = py::cast<py::tuple>(args[i]);
                    if(arg.size() == 3) {
                        args[i] = py::cast(new axis::regular_uoflow(py::cast<unsigned>(arg[0]),
                                                                    py::cast<double>(arg[1]),
                                                                    py::cast<double>(arg[2]),
                                                                    py::str()),
                                           py::return_value_policy::take_ownership);
                    } else if(arg.size() == 4) {
                        try {
                            py::cast<double>(arg[3]);
                            throw py::type_error("The fourth argument (metadata) in the tuple cannot be numeric!");
                        } catch(const py::cast_error &) {
                        }

                        args[i] = py::cast(new axis::regular_uoflow(py::cast<unsigned>(arg[0]),
                                                                    py::cast<double>(arg[1]),
                                                                    py::cast<double>(arg[2]),
                                                                    py::cast<metadata_t>(arg[3])),
                                           py::return_value_policy::take_ownership);
                    } else {
                        throw py::type_error(
                            "Only (bins, start, stop) and (bins, start, stop, metadata) tuples accepted");
                    }
                }
            }

            // We try each possible axes type that has high-performance single-type overloads.

            try {
                return try_cast<storage::int_, storage::atomic_int, storage::unlimited>(
                    storage, [&args](auto &&storage) {
                        auto reg = py::cast<axes::regular_uoflow>(args);
                        return py::cast(bh::make_histogram_with(storage, reg), py::return_value_policy::move);
                    });
            } catch(const py::cast_error &) {
            }

            try {
                return try_cast<storage::int_>(storage, [&args](auto &&storage) {
                    auto reg = py::cast<axes::regular_noflow>(args);
                    return py::cast(bh::make_histogram_with(storage, reg), py::return_value_policy::move);
                });
            } catch(const py::cast_error &) {
            }

            // fallback to slower generic implementation
            auto axes = py::cast<axes::any>(args);

            return try_cast<storage::int_, storage::double_, storage::unlimited, storage::weight, storage::atomic_int>(
                storage, [&axes](auto &&storage) {
                    return py::cast(bh::make_histogram_with(storage, axes), py::return_value_policy::move);
                });
        },
        "Make any histogram");

    // This factory makes a class that can be used to create histograms and also be used in is_instance
    py::object factory_meta_py = py::module::import("boost.histogram_utils").attr("FactoryMeta");

    m.attr("histogram") = factory_meta_py(m.attr("make_histogram"),
                                          py::make_tuple(hist.attr("regular_unlimited"),
                                                         hist.attr("regular_int"),
                                                         hist.attr("regular_atomic_int"),
                                                         hist.attr("regular_noflow_int"),
                                                         hist.attr("any_int"),
                                                         hist.attr("any_atomic_int"),
                                                         hist.attr("any_double"),
                                                         hist.attr("any_unlimited"),
                                                         hist.attr("any_weight")));
}
