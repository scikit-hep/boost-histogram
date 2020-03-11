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
                                   "end={3}{4}, mode={5}))")
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

    using slice_mode = bh::algorithm::slice_mode;

    py::enum_<slice_mode>(algorithm, "slice_mode")
        .value("shrink", slice_mode::shrink)
        .value("crop", slice_mode::crop);

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

        .def("crop_and_rebin",
             py::overload_cast<unsigned, double, double, unsigned>(
                 &bh::algorithm::crop_and_rebin),
             "iaxis"_a,
             "lower"_a,
             "upper"_a,
             "merge"_a,
             "Crop and rebin option to be used in reduce().\n"
             "\n"
             "To crop and rebin in one command. Equivalent to passing both the "
             "crop and the\n"
             "rebin option for the same axis to reduce.\n"
             "\n"
             ":param iaxis: which axis to operate on.\n"
             ":param lower: lowest bound that should be kept.\n"
             ":param upper: highest bound that should be kept. If upper is inside "
             "bin interval, the whole "
             "interval is removed.\n"
             ":param merge: how many adjacent bins to merge into one.")
        .def(
            "crop_and_rebin",
            py::overload_cast<double, double, unsigned>(&bh::algorithm::crop_and_rebin),
            "lower"_a,
            "upper"_a,
            "merge"_a,
            "Positional crop and rebin option to be used in reduce().\n"
            "\n"
            "To crop and rebin in one command. Equivalent to passing both the "
            "crop and the\n"
            "rebin option for the same axis to reduce.\n"
            "\n"
            ":param iaxis: which axis to operate on.\n"
            ":param lower: lowest bound that should be kept.\n"
            ":param upper: highest bound that should be kept. If upper is inside "
            "bin interval, the whole "
            "interval is removed.\n"
            ":param merge: how many adjacent bins to merge into one.")

        .def("slice_and_rebin",
             py::overload_cast<unsigned,
                               bh::axis::index_type,
                               bh::axis::index_type,
                               unsigned,
                               slice_mode>(&bh::algorithm::slice_and_rebin),
             "iaxis"_a,
             "begin"_a,
             "end"_a,
             "merge"_a,
             "mode"_a = slice_mode::shrink,
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
             ":param mode: see slice_mode")
        .def("slice_and_rebin",
             py::overload_cast<bh::axis::index_type,
                               bh::axis::index_type,
                               unsigned,
                               slice_mode>(&bh::algorithm::slice_and_rebin),
             "begin"_a,
             "end"_a,
             "merge"_a,
             "mode"_a = slice_mode::shrink,
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
             ":param mode: see slice_mode")

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

        .def("crop",
             py::overload_cast<unsigned, double, double>(&bh::algorithm::crop),
             "iaxis"_a,
             "lower"_a,
             "upper"_a)
        .def("crop",
             py::overload_cast<double, double>(&bh::algorithm::crop),
             "lower"_a,
             "upper"_a)

        .def("slice",
             py::overload_cast<unsigned,
                               bh::axis::index_type,
                               bh::axis::index_type,
                               slice_mode>(&bh::algorithm::slice),
             "iaxis"_a,
             "begin"_a,
             "end"_a,
             "mode"_a = slice_mode::shrink)
        .def("slice",
             py::overload_cast<bh::axis::index_type, bh::axis::index_type, slice_mode>(
                 &bh::algorithm::slice),
             "begin"_a,
             "end"_a,
             "mode"_a = slice_mode::shrink)

        ;
}
