// Copyright 2015-2018 Hans Dembinski and Henry Schreiner
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/histogram/python/pybind11.hpp>
#include <pybind11/operators.h>

#include <boost/histogram/python/axis.hpp>
#include <boost/histogram/python/storage.hpp>

#include <boost/histogram.hpp>
#include <boost/histogram/axis/ostream.hpp>
#include <boost/histogram/ostream.hpp>

#include <cassert>
#include <vector>
#include <sstream>

#ifndef BOOST_HISTOGRAM_AXIS_LIMIT
#define BOOST_HISTOGRAM_AXIS_LIMIT 16
#endif

namespace bh = boost::histogram;
using namespace boost::histogram::literals;

template<typename A, typename S>
py::class_<bh::histogram<A, S>>&& register_histogram_by_type(py::module& m, const char* name, const char* desc) {
    
    using histogram_t = bh::histogram<A, S>;
    // using in_storage_t = typename bh::histogram<A, S>::value_type; // No unlimited type
    
    py::class_<histogram_t> hist(m, name, desc /*, py::buffer_protocol() */);
    
    hist
    .def(py::init<const A&, S>(), "axes"_a, "storage"_a=S())
    
    /*
    .def_buffer([](histogram_t &h) -> py::buffer_info {
        auto rank = h.rank();
        auto rows = bh::axis::traits::extend(h.axis(0_c));
        auto cols = bh::axis::traits::extend(h.axis(1_c));
        
        return py::buffer_info(
                               &h[0],                                         // Pointer to buffer
                               sizeof(in_storage_t),                          // Size of one scalar
                               py::format_descriptor<in_storage_t>::format(), // Python struct-style format descriptor
                               rank,                                          // Number of dimensions
                               {rows, cols},                                  // Buffer dimensions
                               { sizeof(in_storage_t),                        // Strides (in bytes) for each index
                                 sizeof(in_storage_t) * rows}
                               );
    })
    */
    
    .def("rank", &histogram_t::rank,
         "Number of axes (dimensions) of histogram" )
    .def("size", &histogram_t::size,
         "Total number of bins in the histogram (including underflow/overflow)" )
    .def("reset", &histogram_t::reset,
         "Reset bin counters to zero")
    
    .def(py::self + py::self)
    .def(py::self == py::self)
    .def(py::self != py::self)
    .def(py::self *= double())
    .def(py::self /= double())
    
    .def("axis",
        [](histogram_t &self, unsigned i){return self.axis(i);},
     "Get N-th axis with runtime index")
    
    .def("fill", [](histogram_t &self, py::array_t<double> &data){
        py::buffer_info data_buf = data.request(); // TODO: make const?
        if(self.rank() == 1 && data_buf.shape.size() == 1) {
            const double *ptr1 = (const double *) data_buf.ptr;
            py::gil_scoped_release gil;
            for(size_t i = 0; i<data_buf.shape.at(0); i++)
                self(ptr1[i]);
        } else {
            if(data_buf.ndim > 2)
                throw std::runtime_error("max 2D array required");
            else if(data_buf.ndim < 2 && self.rank() > 1)
                throw std::runtime_error("2D array required for >1D histograms");
            else if(data_buf.shape.at(0) != self.rank())
                throw std::runtime_error("First dimension must match histogram");
        
            auto r = data.unchecked<2>();
            
            py::gil_scoped_release gil;
            for(size_t i=0; i<r.shape(1); i++)
                self(r(0, i), r(1, i));
        }
        
    }, "Add data to histogram, diminsionality must match", "data"_a)
    
    .def("__call__", [](histogram_t &self, py::args &args){
        size_t size = args.size();
        if(size == 1)
            self(py::cast<double>(args[0]));
        else if (size == 2)
            self(py::cast<double>(args[0]), py::cast<double>(args[1]));
        else
            throw py::index_error();
        },
        "Add a value to the historgram")
    
    .def("at", [](histogram_t &self, py::args &args) {
        // Optimize for no dynamic?
        auto int_args = py::cast<std::vector<int>>(args);
        return self.at(int_args);
        },
         "Access bin counter at indices")
    
    .def("__repr__", [](histogram_t &self){
        std::ostringstream out;
        out << self;
        return out.str();
    })
    
    ;
    
    return std::move(hist);
}

void register_histogram(py::module& m) {
    py::module hist = m.def_submodule("hist");
    
    register_histogram_by_type<axes::regular, bh::default_storage>(hist,
        "regular_unlimited",
        "N-dimensional histogram for real-valued data.");
    
    register_histogram_by_type<axes::regular, vector_int_storage>(hist,
        "regular_int",
        "N-dimensional histogram for int-valued data.");
    
    register_histogram_by_type<axes::regular, bh::weight_storage>(hist,
        "regular_weight",
        "N-dimensional histogram for real-valued data with weights.");
    
    register_histogram_by_type<axes::any, vector_int_storage>(hist,
        "any_int",
        "N-dimensional histogram for int-valued data with any axis types.");
    
    register_histogram_by_type<axes::regular_1D, vector_int_storage>(hist,
        "regular_int_1d",
        "1-dimensional histogram for int valued data.");
    
    register_histogram_by_type<axes::regular_2D, vector_int_storage>(hist,
        "regular_int_2d",
        "2-dimensional histogram for int valued data.");
    
}
