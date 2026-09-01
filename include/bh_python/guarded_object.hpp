// Copyright 2026 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

#include <bh_python/pybind11.hpp>

#include <pybind11/pytypes.h>

#include <utility>

/// A py::object member for C++ values that are copied and destroyed with the
/// GIL released (axes during reduce, project, growing fill): every special
/// member that touches reference counts attaches to the interpreter first.
/// Moves and the adopting constructor are refcount-neutral, so GIL-free.
class guarded_object {
    py::object obj_;

  public:
    guarded_object() noexcept = default;

    /// Adopt an existing reference; rvalue-only so no copy (incref) can happen
    /// here, which is why no GIL is needed
    explicit guarded_object(py::object&& obj) noexcept
        : obj_(std::move(obj)) {}

    guarded_object(const guarded_object& other) {
        const py::gil_scoped_acquire gil;
        obj_ = other.obj_;
    }

    guarded_object(guarded_object&&) noexcept = default;

    guarded_object& operator=(const guarded_object& other) {
        if(this != &other) {
            const py::gil_scoped_acquire gil;
            obj_ = other.obj_;
        }
        return *this;
    }

    // Swap defers the decref of the old value to other's guarded destructor
    guarded_object& operator=(guarded_object&& other) noexcept {
        std::swap(obj_, other.obj_);
        return *this;
    }

    // Acquiring the GIL can throw if no thread state can be made; leaking the
    // reference is the only safe option then
    ~guarded_object() {
        if(!obj_)
            return;
        try {
            const py::gil_scoped_acquire gil;
            obj_ = py::object();
        } catch(...) { // NOLINT(bugprone-empty-catch)
        }
    }

    /// Access the held object; the caller must hold the GIL to use or copy it
    const py::object& unguarded_get() const noexcept { return obj_; }

    /// Mutable access; the caller must hold the GIL to assign through this
    py::object& unguarded_ref() noexcept { return obj_; }
};
