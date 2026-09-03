// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <bh_python/guarded_object.hpp>
#include <bh_python/pybind11.hpp>

#include <pybind11/pytypes.h>

#include <utility>

/// Axis metadata: a Python dict, shared with the Python axis wrapper's
/// __dict__. The dict is held through guarded_object because axes are copied
/// and destroyed with the GIL released (reduce, project, growing fill).
class metadata_t {
    guarded_object data_; // a dict; null only after being moved from

  public:
    metadata_t()
        : data_(make_dict()) {}

    /// Adopt an existing reference; rvalue-only, so no GIL needed
    explicit metadata_t(py::object&& data) noexcept
        : data_(std::move(data)) {}

    /// Copying an axis must not alias its dict with the source axis, so make
    /// a fresh shallow copy here rather than sharing guarded_object's
    /// reference. Code that wants the identical dict (e.g. the Python
    /// wrapper's raw_metadata getter) must go through unguarded_obj() instead
    /// of copying a metadata_t.
    metadata_t(const metadata_t& other)
        : data_(shallow_copy_dict(other.data_.unguarded_get())) {}
    metadata_t& operator=(const metadata_t& other) {
        if(this != &other)
            data_ = guarded_object(shallow_copy_dict(other.data_.unguarded_get()));
        return *this;
    }
    metadata_t(metadata_t&&) noexcept            = default;
    metadata_t& operator=(metadata_t&&) noexcept = default;
    ~metadata_t()                                = default;

    /// Access the held dict; hold the GIL to use or copy it
    const py::object& unguarded_obj() const noexcept { return data_.unguarded_get(); }

    // Boost compares axes in a noexcept context, so these must not throw. If
    // the Python comparison raises, or its result has no truth value (a NumPy
    // array), fall back to identity: only the same dict object is equal.
    bool operator==(const metadata_t& other) const noexcept {
        const py::gil_scoped_acquire gil;
        try {
            return unguarded_obj().equal(other.unguarded_obj());
        } catch(...) {
            // pybind11 fetched the Python error into the exception object
            return unguarded_obj().is(other.unguarded_obj());
        }
    }
    bool operator!=(const metadata_t& other) const noexcept {
        return !operator==(other);
    }

  private:
    static py::object make_dict() {
        const py::gil_scoped_acquire gil;
        return py::dict();
    }

    static py::object shallow_copy_dict(const py::object& obj) {
        const py::gil_scoped_acquire gil;
        return py::reinterpret_steal<py::object>(PyDict_Copy(obj.ptr()));
    }
};

/// Deepcopy the held dict, for __deepcopy__ implementations
inline metadata_t deep_copy_metadata(const metadata_t& m, const py::object& memo) {
    py::module const copy = py::module::import("copy");
    return metadata_t{copy.attr("deepcopy")(m.unguarded_obj(), memo)};
}

namespace pybind11 {
namespace detail {
/// Convert to/from the held dict itself, preserving identity: the same dict
/// object is shared between the C++ axis and the Python wrapper's __dict__.
template <>
struct type_caster<metadata_t> {
    PYBIND11_TYPE_CASTER(metadata_t, const_name("dict"));

    // start null instead of allocating a dict that load() would discard
    type_caster()
        : value{object()} {}

    bool load(handle src, bool /*convert*/) {
        if(!isinstance<dict>(src))
            return false;
        value = metadata_t{reinterpret_borrow<object>(src)};
        return true;
    }

    static handle
    cast(const metadata_t& src, return_value_policy /*policy*/, handle /*parent*/) {
        return src.unguarded_obj().inc_ref();
    }
};
} // namespace detail
} // namespace pybind11
