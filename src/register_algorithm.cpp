// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#include <bh_python/pybind11.hpp>

#include <boost/histogram/algorithm/reduce.hpp>

void register_algorithms(py::module& algorithm) {
    py::class_<bh::algorithm::reduce_command>(algorithm, "reduce_command")
        .def("__repr__", [](const bh::algorithm::reduce_command& self) {
            using state_t = bh::algorithm::reduce_command::state_t;

            const char* postfix
                = (self.state == state_t::rebin || self.merge <= 1) ? "" : "_and_rebin";
            const char* start = self.iaxis == bh::algorithm::reduce_command::unset
                                    ? ""
                                    : "iaxis={0}, ";
            const char* merge = (self.state == state_t::rebin || self.merge <= 1)
                                    ? ""
                                    : ", merge={3}";

            std::string name;
            py::str result;

            switch(self.state) {
            case state_t::slice:
                name = std::string("slice") + postfix + "(" + start
                       + "begin={1}, end={2}" + merge + ")";
                result = py::str(name).format(
                    self.iaxis, self.begin.index, self.end.index, self.merge);
                break;
            case state_t::shrink:
                name = std::string("shink") + postfix + "(" + start
                       + "lower={1}, upper={2}" + merge + ")";
                result = py::str(name).format(
                    self.iaxis, self.begin.value, self.end.value, self.merge);
                break;
            case state_t::rebin:
                name   = std::string("rebin") + postfix + "(" + start + "merge={1})";
                result = py::str(name).format(self.iaxis, self.merge);
                break;
            }

            return result;
        });

    py::class_<bh::algorithm::shrink_and_rebin, bh::algorithm::reduce_command>(
        algorithm, "shrink_and_rebin")
        .def(py::init<unsigned, double, double, unsigned>(),
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
        .def(py::init<double, double, unsigned>(),
             "lower"_a,
             "upper"_a,
             "merge"_a,
             "Shortcut form of shrink_and_reduce")

        ;

    py::class_<bh::algorithm::slice_and_rebin, bh::algorithm::reduce_command>(
        algorithm, "slice_and_rebin")
        .def(py::init<unsigned, bh::axis::index_type, bh::axis::index_type, unsigned>(),
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
             ":param merge: how many adjacent bins to merge into one.")
        .def(py::init<bh::axis::index_type, bh::axis::index_type, unsigned>(),
             "begin"_a,
             "end"_a,
             "merge"_a,
             "Shortcut form of slice_and_rebin.");

    py::class_<bh::algorithm::rebin, bh::algorithm::reduce_command>(algorithm, "rebin")
        .def(py::init<unsigned, unsigned>(), "iaxis"_a, "merge"_a)
        .def(py::init<unsigned>(), "merge"_a)

        ;

    py::class_<bh::algorithm::shrink, bh::algorithm::reduce_command>(algorithm,
                                                                     "shrink")
        .def(py::init<unsigned, double, double>(), "iaxis"_a, "lower"_a, "upper"_a)
        .def(py::init<double, double>(), "lower"_a, "upper"_a)

        ;

    py::class_<bh::algorithm::slice, bh::algorithm::reduce_command>(algorithm, "slice")
        .def(py::init<unsigned, bh::axis::index_type, bh::axis::index_type>(),
             "iaxis"_a,
             "begin"_a,
             "end"_a)
        .def(py::init<bh::axis::index_type, bh::axis::index_type>(), "begin"_a, "end"_a)

        ;
}
