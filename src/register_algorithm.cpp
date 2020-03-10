// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#include <bh_python/pybind11.hpp>

#include <boost/histogram/algorithm/reduce.hpp>

void register_algorithms(py::module& algorithm) {
    py::class_<bh::algorithm::reduce_command>(algorithm, "reduce_command")
        .def(py::init<bh::algorithm::reduce_command>())
        .def("__repr__", [](const bh::algorithm::reduce_command& self) {
            using range_t = bh::algorithm::reduce_command::range_t;

            if(self.range != range_t::none) {
                const char* suffix = self.merge > 0 ? "_and_rebin" : "";
                const char* start  = self.iaxis == bh::algorithm::reduce_command::unset
                                        ? ""
                                        : "iaxis={0}, ";
                const char* merge = self.merge > 0 ? ", merge={3}" : "";

                if(self.range == range_t::indices) {
                    return py::str("reduce_command(slice{0}({1}, begin={2}, "
                                   "end={3}{4}, crop={5}))")
                        .format(suffix,
                                start,
                                self.begin.index,
                                self.end.index,
                                merge,
                                self.crop);
                } else {
                    return py::
                        str("reduce_command(shrink{0}({1}, lower={2}, upper={3}{4}))")
                            .format(
                                suffix, start, self.begin.value, self.end.value, merge);
                }
            }

            // self.range == range_t::none
            return py::str("reduce_command(merge({0}))").format(self.merge);
        });

    algorithm
        .def("shrink_and_rebin",
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
             ":param merge: how many adjacent bins to merge into one.")
        .def("shrink_and_rebin",
             py::overload_cast<double, double, unsigned>(
                 &bh::algorithm::shrink_and_rebin),
             "lower"_a,
             "upper"_a,
             "merge"_a,
             "Positional shrink and rebin option to be used in reduce().\n"
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
             ":param merge: how many adjacent bins to merge into one.")

        .def(
            "slice_and_rebin",
            [](unsigned iaxis,
               bh::axis::index_type a,
               bh::axis::index_type b,
               unsigned merge,
               bool crop) {
                return bh::algorithm::slice_and_rebin(
                    iaxis,
                    a,
                    b,
                    merge,
                    crop ? bh::algorithm::slice_mode::crop
                         : bh::algorithm::slice_mode::shrink);
            },
            "iaxis"_a,
            "begin"_a,
            "end"_a,
            "merge"_a,
            "crop"_a = false,
            "Slice and rebin option to be used in reduce().\n"
            "\n"
            "To slice and rebin in one command. Equivalent to passing both the "
            "slice() and the\n"
            "rebin() option for the same axis to reduce.\n"
            "\n"
            ":param iaxis: which axis to operate on.\n"
            ":param begin: first index that should be kept.\n"
            ":param end: one past the last index that should be kept.\n"
            ":param merge: how many adjacent bins to merge into one.\n"
            ":param crop: if false (default), add counts in removed bins to flow bins; "
            "if true, remove counts completely")
        .def(
            "slice_and_rebin",
            [](bh::axis::index_type a,
               bh::axis::index_type b,
               unsigned merge,
               bool crop) {
                return bh::algorithm::slice_and_rebin(
                    a,
                    b,
                    merge,
                    crop ? bh::algorithm::slice_mode::crop
                         : bh::algorithm::slice_mode::shrink);
            },
            "begin"_a,
            "end"_a,
            "merge"_a,
            "crop"_a = false,
            "Positional slice and rebin option to be used in reduce().\n"
            "\n"
            "To slice and rebin in one command. Equivalent to passing both the "
            "slice() and the\n"
            "rebin() option for the same axis to reduce.\n"
            "\n"
            ":param iaxis: which axis to operate on.\n"
            ":param begin: first index that should be kept.\n"
            ":param end: one past the last index that should be kept.\n"
            ":param merge: how many adjacent bins to merge into one.\n"
            ":param crop: if false (default), add counts in removed bins to flow bins; "
            "if true, remove counts completely")

        .def("rebin",
             py::overload_cast<unsigned, unsigned>(&bh::algorithm::rebin),
             "iaxis"_a,
             "merge"_a)
        .def("rebin", py::overload_cast<unsigned>(&bh::algorithm::rebin), "merge"_a)

        .def("shrink",
             py::overload_cast<unsigned, double, double>(&bh::algorithm::shrink),
             "iaxis"_a,
             "lower"_a,
             "upper"_a)
        .def("shrink",
             py::overload_cast<double, double>(&bh::algorithm::shrink),
             "lower"_a,
             "upper"_a)

        .def(
            "slice",
            [](unsigned iaxis,
               bh::axis::index_type a,
               bh::axis::index_type b,
               bool crop) {
                return bh::algorithm::slice(iaxis,
                                            a,
                                            b,
                                            crop ? bh::algorithm::slice_mode::crop
                                                 : bh::algorithm::slice_mode::shrink);
            },
            "iaxis"_a,
            "begin"_a,
            "end"_a,
            "crop"_a = false)
        .def(
            "slice",
            [](bh::axis::index_type a, bh::axis::index_type b, bool crop) {
                return bh::algorithm::slice(a,
                                            b,
                                            crop ? bh::algorithm::slice_mode::crop
                                                 : bh::algorithm::slice_mode::shrink);
            },
            "begin"_a,
            "end"_a,
            "crop"_a = false)

        ;
}
