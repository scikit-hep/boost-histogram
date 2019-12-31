// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#include <bh_python/pybind11.hpp>

#include <boost/histogram/algorithm/reduce.hpp>

void register_algorithms(py::module& algorithm) {
    py::class_<bh::algorithm::reduce_option>(algorithm, "_reduce_option")
        .def(py::init<unsigned,
                      bool,
                      bh::axis::index_type,
                      bh::axis::index_type,
                      bool,
                      double,
                      double,
                      unsigned>(),
             "iaxis"_a,
             "indices_set"_a,
             "begin"_a,
             "end"_a,
             "values_set"_a,
             "lower"_a,
             "upper"_a,
             "merge"_a,
             "Constructor for reduce_option; use creation functions instead.")
        .def("__repr__", [](const bh::algorithm::reduce_option& self) {
            return py::str("reduce_option(iaxis={}, indices_set={}, begin={}, end={}, "
                           "values_set={}, lower={}, "
                           "upper={}, merge={})")
                .format(self.iaxis,
                        self.indices_set,
                        self.begin,
                        self.end,
                        self.values_set,
                        self.lower,
                        self.upper,
                        self.merge);
        });

    algorithm.def("shrink_and_rebin",
                  py::overload_cast<unsigned, double, double, unsigned>(
                      &bh::algorithm::shrink_and_rebin),
                  "iaxis"_a,
                  "lower"_a,
                  "upper"_a,
                  "merge"_a,
                  "Shrink and rebin option to be used in reduce().\n"
                  "\n"
                  "To shrink and rebin in one command. Equivalent to passing both the "
                  "shrink() and the\n"
                  "rebin() option for the same axis to reduce.\n"
                  "\n"
                  ":param iaxis: which axis to operate on.\n"
                  ":param lower: lowest bound that should be kept.\n"
                  ":param upper: highest bound that should be kept. If upper is inside "
                  "bin interval, the whole "
                  "interval is removed.\n"
                  ":param merge: how many adjacent bins to merge into one.");

    algorithm.def(
        "shrink_and_rebin",
        py::overload_cast<double, double, unsigned>(&bh::algorithm::shrink_and_rebin),
        "lower"_a,
        "upper"_a,
        "merge"_a);

    algorithm.def("slice_and_rebin",
                  py::overload_cast<unsigned,
                                    bh::axis::index_type,
                                    bh::axis::index_type,
                                    unsigned>(&bh::algorithm::slice_and_rebin),
                  "iaxis"_a,
                  "begin"_a,
                  "end"_a,
                  "merge"_a,
                  "Slice and rebin option to be used in reduce().\n"
                  "\n"
                  "To slice and rebin in one command. Equivalent to passing both the "
                  "slice() and the\n"
                  "rebin() option for the same axis to reduce.\n"
                  "\n"
                  ":param iaxis: which axis to operate on.\n"
                  ":param begin: first index that should be kept.\n"
                  ":param end: one past the last index that should be kept.\n"
                  ":param merge: how many adjacent bins to merge into one.");

    algorithm.def(
        "slice_and_rebin",
        py::overload_cast<bh::axis::index_type, bh::axis::index_type, unsigned>(
            &bh::algorithm::slice_and_rebin),
        "begin"_a,
        "end"_a,
        "merge"_a);

    algorithm.def("rebin",
                  py::overload_cast<unsigned, unsigned>(&bh::algorithm::rebin),
                  "iaxis"_a,
                  "merge"_a);

    algorithm.def(
        "rebin", py::overload_cast<unsigned>(&bh::algorithm::rebin), "merge"_a);

    algorithm.def("shrink",
                  py::overload_cast<unsigned, double, double>(&bh::algorithm::shrink),
                  "iaxis"_a,
                  "lower"_a,
                  "upper"_a);

    algorithm.def("shrink",
                  py::overload_cast<double, double>(&bh::algorithm::shrink),
                  "lower"_a,
                  "upper"_a);

    algorithm.def(
        "slice",
        py::overload_cast<unsigned, bh::axis::index_type, bh::axis::index_type>(
            &bh::algorithm::slice),
        "iaxis"_a,
        "begin"_a,
        "end"_a);

    algorithm.def("slice",
                  py::overload_cast<bh::axis::index_type, bh::axis::index_type>(
                      &bh::algorithm::slice),
                  "begin"_a,
                  "end"_a);
}
