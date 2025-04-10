from __future__ import annotations

import copy
from typing import Any, ClassVar, TypeVar

import boost_histogram

from .._compat.typing import Self
from .._core import axis as ca
from .._utils import register

T = TypeVar("T", bound="AxisTransform")

__all__ = ["AxisTransform", "Function", "Pow", "log", "sqrt"]


def __dir__() -> list[str]:
    return __all__


class AxisTransform:
    __slots__ = ("_this",)
    _family: ClassVar[object]  # pylint: disable=declare-non-slot
    _this: ca.transform._BaseTransform

    def __init_subclass__(cls, *, family: object) -> None:
        super().__init_subclass__()
        cls._family = family

    def __copy__(self) -> Self:
        other: Self = self.__class__.__new__(self.__class__)
        other._this = copy.copy(self._this)
        return other

    @classmethod
    def _convert_cpp(cls, this: Any) -> Self:
        self: Self = cls.__new__(cls)
        self._this = this
        return self

    def __repr__(self) -> str:
        if hasattr(self, "_this"):
            return repr(self._this)

        return f"{self.__class__.__name__}() # Missing _this, broken class"

    def _produce(self, bins: int, start: float, stop: float) -> Any:
        raise NotImplementedError()

    def __init__(self) -> None:
        "Create a new transform instance"
        raise NotImplementedError()

    def forward(self, value: float) -> float:
        "Compute the forward transform"
        return self._this.forward(value)

    def inverse(self, value: float) -> float:
        "Compute the inverse transform"
        return self._this.inverse(value)


@register({ca.transform.pow})
class Pow(AxisTransform, family=boost_histogram):
    __slots__ = ()
    _type = ca.regular_pow
    _this: ca.transform.pow

    # Note: this comes from family
    _types: ClassVar[set[type[ca.transform.pow]]]

    def __init__(self, power: float):  # pylint: disable=super-init-not-called
        "Create a new transform instance"
        (cpp_class,) = self._types
        self._this = cpp_class(power)

    @property
    def power(self) -> float:
        "The power of the transform"
        return self._this.power

    # This one does need to be a normal method
    def _produce(self, bins: int, start: float, stop: float) -> Any:
        return self.__class__._type(bins, start, stop, self.power)


@register({ca.transform.func_transform})
class Function(AxisTransform, family=boost_histogram):
    __slots__ = ()
    _type = ca.regular_trans
    _this: ca.transform.func_transform

    # Note: this comes from family
    _types: ClassVar[set[type[ca.transform.func_transform]]]

    def __init__(  # pylint: disable=super-init-not-called
        self, forward: Any, inverse: Any, *, convert: Any = None, name: str = ""
    ):
        """
        Create a functional transform from a ctypes double(double) function
        pointer or any object that provides such an interface through a
        ``.ctypes`` attribute (such as numba.cfunc). A pure python function *can*
        be adapted to a ctypes pointer, but please use a Variable axis instead or
        use something like numba to produce a compiled function pointer. You can
        manually specify the repr name with ``name=``.

        Example of Numba use:
        ---------------------

            @numba.cfunc(numba.float64(numba.float64,))
            def exp(x):
                return math.exp(x)

            @numba.cfunc(numba.float64(numba.float64,))
            def log(x):
                return math.log(x)

        Example of slow CTypes use:
        ---------------------------

            ftype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
            log = ftype(math.log)
            exp = ftype(math.exp)


        Now you can supply these functions, and you will get a high performance
        transformation axis.

        You can also supply an optional conversion function; this will take the input
        forward and inverse and call them before producing a transform. This enables
        pickling, as well, since ctypes pointers are not picklable. A few common
        utilities have been supplied:

        * ``convert.numba``: Compile using numba (required)
        * ``convert.python``: Just call the Python function (15-90x slower than compiled)

        See also
        --------

        * ``Numbify(forward, inverse, *, name='')``: Uses convert=convert.numba
        * ``PythonFunction(forward, inverse, *, name='')``: Uses convert=convert.python

        """

        (cpp_class,) = self._types
        self._this = cpp_class(forward, inverse, convert, name)

    # This one does need to be a normal method
    def _produce(self, bins: int, start: float, stop: float) -> Any:
        return self.__class__._type(bins, start, stop, self._this)


def _internal_conversion(name: str) -> Any:
    return getattr(ca.transform, name)


sqrt = Function("_sqrt_fn", "_sq_fn", convert=_internal_conversion, name="sqrt")
log = Function("_log_fn", "_exp_fn", convert=_internal_conversion, name="log")
