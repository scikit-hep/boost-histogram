// Copyright 2018-2019 Henry Schreiner and Hans Dembinski
//
// Distributed under the 3-Clause BSD License.  See accompanying
// file LICENSE or https://github.com/scikit-hep/boost-histogram for details.

#pragma once

inline bool PyObject_Check(void *value) { return value != nullptr; }

struct metadata_t : py::object {
    PYBIND11_OBJECT_DEFAULT(metadata_t, object, PyObject_Check);

    metadata_t(const metadata_t &) = default;
    metadata_t(metadata_t &&)      = default;
    metadata_t &operator=(const metadata_t &) = default;
    metadata_t &operator=(metadata_t &&) = default;

    metadata_t &operator=(const object &obj) {
        object::operator=(obj);
        return *this;
    }
    metadata_t &operator=(py::object &&obj) {
        object::operator=(std::move(obj));
        return *this;
    }

    bool operator==(const metadata_t &other) const { return py::object::equal(other); }
    bool operator!=(const metadata_t &other) const {
        return py::object::not_equal(other);
    }

    template <class Archive>
    void serialize(Archive &ar, unsigned /* version */) {
        ar &static_cast<py::object &>(*this);
    }
};
