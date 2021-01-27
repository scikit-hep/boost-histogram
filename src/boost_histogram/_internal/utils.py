# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

del absolute_import, division, print_function

# Custom families (other packages can define custom families)
MAIN_FAMILY = object()  # This family will be used as a fallback

# These are not exported because custom user classes do not need to
# add to the original families, they should make their own.


def set_module(name):
    """
    Set the __module__ attribute on a class. Very
    similar to numpy.core.overrides.set_module.
    """

    def add_module(cls):
        cls.__module__ = name
        return cls

    return add_module


def set_family(family):
    """
    Decorator to set the family of a class. When an object
    is produced from the C++ bindings, it will look through
    all subclasses of the base class (Axis, Transform, Storage,
    etc.) to find a match. If possible, it will match the family
    of the object that is producing it (Histogram, Axis, etc.).
    If a family is not found, it will return the main family
    as a fallback.
    """

    def add_family(cls):
        cls._family = family
        return cls

    return add_family


def register(cpp_types=None):
    """
    Decorator to register a C++ type to a Python class.
    Each class given will be added to a lookup list "_types"
    that cast knows about. It should also part of a "family",
    and any class in a family will cast to the same family.
    See set_family. You do not need to register a class if it
    inherits from the C++ class.

    For example, internally this call:

        ax = hist._axis(0)

    which will get a raw C++ object and need to cast it to a Python
    wrapped object. There is currently one candidates (users
    could add more): MAIN_FAMILY. Cast will use the
    parent class's family to return the correct family. If the
    requested family is not found, then the regular family is the
    fallback.

    This decorator, like other decorators in boost-histogram,
    is safe for pickling since it does not replace the
    original class.

    If nothing or an empty set is passed, this will ensure that this
    class is not selected during the cast process. This can be
    used for simple renamed classes that inject warnings, etc.
    """

    def add_registration(cls):
        if cpp_types is None or len(cpp_types) == 0:
            cls._types = set()
            return cls

        if not hasattr(cls, "_types"):
            cls._types = set()

        for cpp_type in cpp_types:
            if cpp_type in cls._types:
                raise TypeError("You are trying to register {} again".format(cpp_type))

            cls._types.add(cpp_type)

        return cls

    return add_registration


def _cast_make_object(canidate_class, cpp_object, is_class):
    "Make an object for cast"
    if is_class:
        return canidate_class

    elif hasattr(canidate_class, "_convert_cpp"):
        return canidate_class._convert_cpp(cpp_object)

    # Casting down does not work in pybind11,
    # see https://github.com/pybind/pybind11/issues/1640
    # so for now, all non-copy classes must have a
    # _convert_cpp method.

    else:
        return canidate_class(cpp_object)


def cast(self, cpp_object, parent_class):
    """
    This converts a C++ object into a Python object.
    This takes the parent object, the C++ object,
    the Python class. If a class is passed in instead of
    an object, this will return a class instead. The parent
    object (self) can be either a registered class or an
    instance of a registered class.

    Instances simply have their class replaced.

    If a class does not support direction conversion in
    the constructor, it should have _convert_cpp class
    method instead.

    Example:

        cast(self, hist.cpp_axis(), Axis)
        # -> returns Regular(...) if regular axis, etc.

    If self is None, just use the MAIN_FAMILY.
    """
    if self is None:
        family = MAIN_FAMILY
    else:
        family = self._family

    # Convert objects to classes, and remember if we did so
    if isinstance(cpp_object, type):
        is_class = True
        cpp_class = cpp_object
    else:
        is_class = False
        cpp_class = cpp_object.__class__

    # Remember the fallback class if a class in the same family does not exist
    fallback_class = None

    for canidate_class in _walk_subclasses(parent_class):
        # If a class was registered with this c++ type
        if hasattr(canidate_class, "_types"):
            is_valid_type = cpp_class in canidate_class._types
        else:
            is_valid_type = cpp_class in set(_walk_bases(canidate_class))

        if is_valid_type and hasattr(canidate_class, "_family"):
            # Return immediately if the family is right
            if canidate_class._family is family:
                return _cast_make_object(canidate_class, cpp_object, is_class)

            # Or remember the class if it was from the main family
            if canidate_class._family is MAIN_FAMILY:
                fallback_class = canidate_class

    # If no perfect match was registered, return the main family
    if fallback_class is not None:
        return _cast_make_object(fallback_class, cpp_object, is_class)

    raise TypeError(
        "No conversion to {} from {} found.".format(parent_class.__name__, cpp_object)
    )


def _walk_bases(cls):
    for base in cls.__bases__:
        for inner in _walk_bases(base):
            yield inner
        yield base


def _walk_subclasses(cls):
    for base in cls.__subclasses__():
        # Find the furthest child to allow
        # user subclasses to work
        for inner in _walk_subclasses(base):
            yield inner
        yield base
