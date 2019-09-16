// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#include <boost/histogram/python/shared_histogram.hpp>

std::tuple<std::vector<bh::algorithm::reduce_option>, std::vector<unsigned>>
get_slices(py::tuple indexes,
           std::function<bh::axis::index_type(bh::axis::index_type, double)> index_self,
           std::function<bh::axis::index_type(bh::axis::index_type)> size_self) {
    std::tuple<std::vector<bh::algorithm::reduce_option>, std::vector<unsigned>> mytuple;
    std::vector<bh::algorithm::reduce_option> &slices = std::get<0>(mytuple);
    std::vector<unsigned> &projections                = std::get<1>(mytuple);

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
            unsigned merge = 1;
            if(step.is_none()) {
                merge = 1;
            } else if(!py::hasattr(step, "projection")) {
                throw std::out_of_range("The third argument to a slice must be rebin or projection");
            } else if(py::cast<bool>(step.attr("projection")) == true) {
                projections.emplace_back(i);
                if(start.is_none() && stop.is_none())
                    continue;
                else
                    throw std::out_of_range("Currently cut projection is not supported");
            } else {
                if(!py::hasattr(step, "factor"))
                    throw std::out_of_range("Invalid rebin, must have .factor set to an integer");
                merge = py::cast<unsigned>(step.attr("factor"));
            }

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

            slices.push_back(bh::algorithm::slice_and_rebin(static_cast<unsigned>(i), begin, end, merge));
        }
    }

    return mytuple;
}

py::list expand_ellipsis(py::list indexes, py::size_t rank) {
    py::size_t ellipis_index = 0;
    bool ellipsis_found      = false;
    for(py::size_t i = 0; i < indexes.size(); i++) {
        if(py::cast<std::string>(indexes[i].attr("__class__").attr("__name__")) == "ellipsis") {
            if(ellipsis_found)
                throw std::out_of_range("an index can only have a single ellipsis ('...')");
            ellipsis_found = true;
            ellipis_index  = i;
        }
    }

    if(ellipsis_found) {
        if(indexes.size() > rank + 1)
            throw std::out_of_range("IndexError: too many indices for histogram");
        py::size_t additional = rank + 1 - indexes.size();

        py::list new_list;

        // The first part of the list should identical, up to the ellipsis
        for(py::size_t i = 0; i < ellipis_index; i++)
            new_list.append(indexes[i]);

        // Fill out the ellipsis with empty slices
        // py::object builtins = py::import( ? "" : "")
        py::dict builtins = py::cast<py::dict>(py::handle(PyEval_GetBuiltins()));
        for(py::size_t i = 0; i < additional; i++)
            new_list.append(builtins["slice"](py::none()));

        // Add the parts after the ellipsis
        for(py::size_t i = ellipis_index + 1; i < indexes.size(); i++)
            new_list.append(indexes[i]);

        return new_list;
    } else {
        return indexes;
    }
}
