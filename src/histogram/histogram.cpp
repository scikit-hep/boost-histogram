// Copyright 2015-2018 Hans Dembinski and Henry Schreiner
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/histogram/python/pybind11.hpp>
#include <pybind11/operators.h>

#include <boost/histogram/python/axis.hpp>
#include <boost/histogram/python/storage.hpp>
#include <boost/histogram/python/histogram.hpp>

#include <boost/histogram.hpp>
#include <boost/histogram/axis/ostream.hpp>
#include <boost/histogram/ostream.hpp>

#include <boost/mp11.hpp>

#include <cassert>
#include <vector>
#include <sstream>
#include <tuple>
#include <cmath>

namespace bh = boost::histogram;

template <class Histogram>
struct fill_helper {
  fill_helper(Histogram& h, py::args args) : hist(h) {
    auto dim = args.size();
    if (dim == 0)
      throw std::invalid_argument("at least one argument required");

    arrs.reserve(dim);
    for (std::size_t i = 0; i < dim; ++i) {
      auto a = py::cast<py::array_t<double>>(args[i]);
      if (a.ndim() > 1)
        throw std::invalid_argument("array dim must be 0 or 1");
      arrs.emplace_back(a, a.data());
    }

    size = arrs[0].first.size();
    for (std::size_t i = 1; i < dim; ++i)
      if (arrs[i].first.size() != size)
        throw std::invalid_argument("arrays must have same length");
    if (size == 0) // handle scalars by setting size to 1
      ++size;
  }

  // keep function small to minimize code bloat, it is instantiated 32 times :(
  // TODO: solve this more efficiently on the lower Boost::Histogram level
  template <class N>
  void operator()(N) {
    using namespace boost::mp11;
    // N is a compile-time number with N == arrs.size()
    mp_repeat<std::tuple<double>, N> tp;
    // TODO: only release gil when storage is thread-safe
    // py::gil_scoped_release gil;
    for (std::size_t i = 0; i < size; ++i) {
      mp_for_each<mp_iota<N>>([&](auto I) {
        // I is mp_size_t<0>, mp_size_t<1>, ..., mp_size_t<N-1>
        std::get<I>(tp) = arrs[I].second[i];
      });
      hist(tp); // throws invalid_argument if tup has wrong size      
    }
  }

  // specialization for N=0 to prevent compile-time error in histogram
  void operator()(boost::mp11::mp_size_t<0>) {
    throw std::invalid_argument("at least one argument required");
  }

  // specialization for N=1, implement potential optimizations here
  void operator()(boost::mp11::mp_size_t<1>) {
    // compilers often emit faster code for range-based for loops
    struct span {
      const double* begin() const { return ptr; }
      const double* end() const { return ptr + size; }
      const double* ptr;
      std::size_t size;
    };
    // TODO: only release gil when storage is thread-safe
    // py::gil_scoped_release gil;
    for (double xi : span{arrs.front().second, size})
      hist(xi); // throws invalid_argument if hist.rank() != 1 
  }

  Histogram& hist;
  std::vector<std::pair<py::array_t<double>, const double*>> arrs;
  std::size_t size;
};

template<typename A, typename S>
py::class_<bh::histogram<A, S>>&& register_histogram_by_type(py::module& m, const char* name, const char* desc) {
    
    using histogram_t = bh::histogram<A, S>;
   
    py::class_<histogram_t> hist(m, name, desc, py::buffer_protocol());
    
    hist
    .def(py::init<const A&, S>(), "axes"_a, "storage"_a=S())
    
    .def_buffer([](bh::histogram<A, S>& h) -> py::buffer_info
        {return make_buffer(h);})
    
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

    // generic fill for 1 to N args
    .def("__call__",
      [](histogram_t &self, py::args args) {
        boost::mp11::mp_with_index<BOOST_HISTOGRAM_DETAIL_AXES_LIMIT>(args.size(), fill_helper<histogram_t>(self, args));
      }, "Insert data into histogram")

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
    
    register_histogram_by_type<axes::regular_noflow, vector_int_storage>(hist,
        "regular_int_noflow",
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
    
    register_histogram_by_type<axes::regular_noflow_1D, vector_int_storage>(hist,
        "regular_int_noflow_1d",
        "1-dimensional histogram for int valued data.");
    
    register_histogram_by_type<axes::regular_noflow_2D, vector_int_storage>(hist,
        "regular_int_noflow_2d",
        "2-dimensional histogram for int valued data.");
    
}
