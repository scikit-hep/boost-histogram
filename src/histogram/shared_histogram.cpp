// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#include <boost/histogram/python/shared_histogram.hpp>

std::vector<bh::algorithm::reduce_option>
get_slices(py::tuple indexes,
           std::function<bh::axis::index_type(bh::axis::index_type, double)> index_self,
           std::function<bh::axis::index_type(bh::axis::index_type)> size_self) {
    std::vector<bh::algorithm::reduce_option> slices;
    for(bh::axis::index_type i = 0; static_cast<unsigned>(i) < indexes.size(); i++) {
        auto ind = indexes[static_cast<pybind11::size_t>(i)];
        if(!py::isinstance<py::slice>(ind))
            throw std::out_of_range("Invalid arguments as an index, use all integers or all slices, and do not mix");

        py::object start = ind.attr("start");
        py::object stop  = ind.attr("stop");
        py::object step  = ind.attr("step");

        // : means take all from axis
        if(start.is_none() && stop.is_none() && step.is_none()) {
            continue;
        } else {
            // Start can be none, integer, or loc(double)
            bh::axis::index_type begin
                = start.is_none() ? 0
                                  : (py::hasattr(start, "value") ? index_self(i, py::cast<double>(start.attr("value")))
                                                                 : py::cast<bh::axis::index_type>(start));

            // Stop can be none, integer, or loc(double)
            bh::axis::index_type end
                = stop.is_none() ? size_self(i)
                                 : (py::hasattr(stop, "value") ? index_self(i, py::cast<double>(stop.attr("value")))
                                                               : py::cast<bh::axis::index_type>(stop));

            unsigned merge = 1;
            if(step.is_none()) {
                merge = 1;
            } else if(!py::hasattr(step, "projection")) {
                throw std::out_of_range("The third argument to a slice must be rebin or projection");
            } else if(py::cast<bool>(step.attr("projection")) == true) {
                throw std::out_of_range("Currently projection is not supported");
            } else {
                if(!py::hasattr(step, "factor"))
                    throw std::out_of_range("Invalid rebin, must have .factor set to an integer");
                merge = py::cast<unsigned>(step.attr("factor"));
            }

            slices.push_back(bh::algorithm::slice_and_rebin(static_cast<unsigned>(i), begin, end, merge));
        }
    }

    return slices;
}
