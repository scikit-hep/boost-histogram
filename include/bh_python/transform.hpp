// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <bh_python/pybind11.hpp>

#include <boost/core/nvp.hpp>
#include <boost/histogram/axis/regular.hpp>
#include <utility>

#include <pybind11/functional.h>

namespace bh = boost::histogram;

struct func_transform {
    using raw_t = double(double);

    raw_t* _forward = nullptr;
    raw_t* _inverse = nullptr;
    py::object _forward_ob; // Held for reference counting, repr, and pickling
    py::object _inverse_ob;
    py::object _forward_converted; // Held for reference counting if conversion makes a
                                   // new object (ctypes does not bump the refcount)
    py::object _inverse_converted;
    py::object _convert_ob; // Called before computing tranform if not None
    py::str _name;          // Optional name (uses repr from objects otherwise)

    /// Convert an object into a std::function. Can handle ctypes
    /// function pointers and pybind11 C++ functions, or anything
    /// else with a defined convert function
    std::tuple<raw_t*, py::object> compute(py::object& input) {
        // Run the conversion function on the input (unless conversion is None)
        py::object tmp_src = _convert_ob.is_none() ? input : _convert_ob(input);

        // If a CTypes object is present, just use that (numba, for example)
        py::object src = py::getattr(tmp_src, "ctypes", tmp_src);

        // import ctypes
        py::module ctypes = py::module::import("ctypes");

        // Get the type: double(double)
        // function_type = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
        py::handle CFUNCTYPE     = ctypes.attr("CFUNCTYPE");
        py::handle c_double      = ctypes.attr("c_double");
        py::object function_type = CFUNCTYPE(c_double, c_double);

        if(py::isinstance(src, function_type)) {
            py::handle cast     = ctypes.attr("cast");
            py::handle c_void_p = ctypes.attr("c_void_p");

            // ctypes.cast(in, ctypes.c_void_p).value
            py::object addr_obj = cast(src, c_void_p);
            auto addr           = py::cast<std::uintptr_t>(addr_obj.attr("value"));
            auto ptr            = reinterpret_cast<raw_t*>(addr);
            return std::make_tuple(ptr, src);
        }

        // If we made it to this point, we probably have a C++ pybind object or an
        // invalid object. The following is based on the std::function conversion in
        // pybind11/functional.hpp
        if(!py::isinstance<py::function>(src))
            throw py::type_error("Only ctypes double(double) and C++ functions allowed "
                                 "(must be function)");

        auto func = py::reinterpret_borrow<py::function>(src);

        if(auto cfunc = func.cpp_function()) {
            auto c = py::reinterpret_borrow<py::capsule>(
                PyCFunction_GET_SELF(cfunc.ptr()));

            // NOLINTNEXTLINE(google-readability-casting)
            auto rec = (py::detail::function_record*)(c);

            if(rec && rec->is_stateless
               && py::detail::same_type(
                   typeid(raw_t*),
                   *reinterpret_cast<const std::type_info*>(rec->data[1]))) {
                struct capture {
                    raw_t* f;
                };
                return std::make_tuple((reinterpret_cast<capture*>(&rec->data))->f,
                                       src);
            }

            // Note that each error is slighly different just to help with debugging
            throw py::type_error("Only ctypes double(double) and C++ functions allowed "
                                 "(must be stateless)");
        }

        throw py::type_error("Only ctypes double(double) and C++ functions allowed "
                             "(must be cpp function)");
    }

    func_transform(py::object f, py::object i, py::object c, py::str n)
        : _forward_ob(f)
        , _inverse_ob(i)
        , _convert_ob(std::move(c))
        , _name(std::move(n)) {
        std::tie(_forward, _forward_converted) = compute(f);
        std::tie(_inverse, _inverse_converted) = compute(i);
    }

    func_transform()                          = default;
    ~func_transform()                         = default;
    func_transform(const func_transform&)     = default;
    func_transform(func_transform&&) noexcept = default;

    func_transform& operator=(const func_transform&) = default;
    func_transform& operator=(func_transform&&) noexcept = default;

    double forward(double x) const { return _forward(x); }

    double inverse(double x) const { return _inverse(x); }

    bool operator==(const func_transform& other) const noexcept {
        return _forward_ob.equal(other._forward_ob)
               && _inverse_ob.equal(other._inverse_ob);
    }

    template <class Archive>
    void serialize(Archive& ar, unsigned /* version */) {
        ar& boost::make_nvp("forward", _forward_ob);
        ar& boost::make_nvp("inverse", _inverse_ob);
        ar& boost::make_nvp("convert", _convert_ob);
        ar& boost::make_nvp("name", _name);

        if(Archive::is_loading::value) {
            std::tie(_forward, _forward_converted) = compute(_forward_ob);
            std::tie(_inverse, _inverse_converted) = compute(_inverse_ob);
        }
    }
};

namespace boost {
namespace histogram {
namespace detail {
inline const char* axis_suffix(const ::func_transform&) { return "_trans"; }
} // namespace detail
} // namespace histogram
} // namespace boost

/// Simple deep copy for any class *without* a python component
template <class T>
T deep_copy(const T& input, py::object) {
    return T(input);
}

/// Specialization for the case where Python components are present
/// (Function transform in this case)
template <>
inline func_transform deep_copy<func_transform>(const func_transform& input,
                                                py::object memo) {
    py::module copy = py::module::import("copy");

    py::object forward = copy.attr("deepcopy")(input._forward_ob, memo);
    py::object inverse = copy.attr("deepcopy")(input._inverse_ob, memo);
    py::object convert = copy.attr("deepcopy")(input._convert_ob, memo);
    py::str name       = copy.attr("deepcopy")(input._name, memo);

    return func_transform(forward, inverse, convert, name);
}

// Print in repr
template <class CharT, class Traits>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os,
                                              const func_transform&) {
    return os << "func_transform";
}
