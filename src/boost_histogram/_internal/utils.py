import typing
from typing import Callable, ClassVar, Iterator, Optional, Set, Type, TypeVar

import boost_histogram

from .typing import Protocol


class Registerable(Protocol):
    _types: ClassVar[Set[Type[object]]]


class HasFamily(Protocol):
    _family: ClassVar[object]


T = TypeVar("T")


def set_module(name: str) -> Callable[[Type[T]], Type[T]]:
    """
    Set the __module__ attribute on a class. Very
    similar to numpy.core.overrides.set_module.
    """

    def add_module(cls: Type[T]) -> Type[T]:
        cls.__module__ = name
        return cls

    return add_module


def register(
    cpp_types: Optional[Set[Type[object]]] = None,
) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to register a C++ type to a Python class.
    Each class given will be added to a lookup list "_types"
    that cast knows about. It should also part of a "family",
    and any class in a family will cast to the same family.
    You do not need to register a class if it inherits from
    the C++ class.

    For example, internally this call:

        ax = hist._axis(0)

    which will get a raw C++ object and need to cast it to a Python
    wrapped object. There is currently one candidate (users
    could add more): boost_histogram. Cast will use the
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

    def add_registration(cls: Type[T]) -> Type[T]:
        if cpp_types is None or len(cpp_types) == 0:
            cls._types = set()  # type: ignore
            return cls

        if not hasattr(cls, "_types"):
            cls._types = set()  # type: ignore

        for cpp_type in cpp_types:
            if cpp_type in cls._types:  # type: ignore
                raise TypeError(f"You are trying to register {cpp_type} again")

            cls._types.add(cpp_type)  # type: ignore

        return cls

    return add_registration


# TODO: Split into is_class/not is class versions (better typing)
def _cast_make_object(canidate_class: T, cpp_object: object, is_class: bool) -> T:
    "Make an object for cast"
    if is_class:
        return canidate_class

    elif hasattr(canidate_class, "_convert_cpp"):
        return canidate_class._convert_cpp(cpp_object)  # type: ignore

    # Casting down does not work in pybind11,
    # see https://github.com/pybind/pybind11/issues/1640
    # so for now, all non-copy classes must have a
    # _convert_cpp method.

    else:
        return canidate_class(cpp_object)  # type: ignore


def cast(self: object, cpp_object: object, parent_class: Type[T]) -> T:
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

    If self is None, just use the boost_histogram family.
    """
    if self is None:
        family: object = boost_histogram
    else:
        family = self._family  # type: ignore

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
            ccr = typing.cast(Registerable, canidate_class)
            is_valid_type = cpp_class in ccr._types
        else:
            is_valid_type = cpp_class in set(_walk_bases(canidate_class))

        if is_valid_type and hasattr(canidate_class, "_family"):
            ccf = typing.cast(HasFamily, canidate_class)

            # Return immediately if the family is right
            if ccf._family is family:
                return _cast_make_object(ccf, cpp_object, is_class)  # type: ignore

            # Or remember the class if it was from the main family
            if ccf._family is boost_histogram:
                fallback_class = ccf

    # If no perfect match was registered, return the main family
    if fallback_class is not None:
        return _cast_make_object(fallback_class, cpp_object, is_class)  # type: ignore

    raise TypeError(
        f"No conversion to {parent_class.__name__} from {cpp_object} found."
    )


def _walk_bases(cls: Type[object]) -> Iterator[Type[object]]:
    for base in cls.__bases__:
        yield from _walk_bases(base)
        yield base


def _walk_subclasses(cls: Type[object]) -> Iterator[Type[object]]:
    for base in cls.__subclasses__():
        # Find the furthest child to allow
        # user subclasses to work
        yield from _walk_subclasses(base)
        yield base
