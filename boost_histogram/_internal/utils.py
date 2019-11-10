from __future__ import absolute_import, division, print_function

del absolute_import, division, print_function


def cpp_module(cls):
    """
    Simple decorator to declare a class to be in the cpp
    module.
    """
    cls._cpp_mode = True
    return cls


def register(*args):
    """
    Decorator to register a C++ type to a Python class.
    Each class given will be added to a lookup list "_types"
    that cast knows about.

    This decorator, like other decorators in boost-histogram,
    is safe for pickling since it does not replace the
    original class.
    """

    def add_registration(cls):
        if not hasattr(cls, "_types"):
            cls._types = set()
        for cpp_type in args:
            if cpp_type in cls._types:
                raise TypeError("You are trying to register {} again".format(cpp_type))

            cls._types.add(cpp_type)
            if not hasattr(cls, "_cpp_mode"):
                cls._cpp_mode = False
            return cls

    return add_registration


def cast(cpp_object, parent_class, cpp=False):
    """
    This converts a C++ object into a Python object.
    This takes the parent Python class, and an optional
    base parameter, which will only return classes that
    are in the base module.

    If a class does not support direction conversion in
    the constructor, it should have _convert_cpp class
    method instead.

    cpp setting must match the register setting.
    """

    for canidate_class in _walk_subclasses(parent_class):
        print(canidate_class)
        if (
            hasattr(canidate_class, "_types")
            and cpp_object.__class__ in canidate_class._types
            and canidate_class._cpp_mode == cpp
        ):
            print("Making")
            if hasattr(canidate_class, "_convert_cpp"):
                return canidate_class._convert_cpp(cpp_object)
            else:
                return canidate_class(cpp_object)
    raise TypeError(
        "No conversion to {0} from {1} found.".format(
            parent_class.__name__, cpp_object.__class__.__name__
        )
    )


def _walk_subclasses(cls):
    for base in cls.__subclasses__():
        # Find the furthest child to allow
        # user subclasses to work
        for inner in _walk_subclasses(base):
            yield inner
        yield base
