from functools import partial
from typing import Any, List, Tuple, TypeVar

import numpy as np

from .axis import Axis
from .utils import set_module

A = TypeVar("A", bound="ArrayTuple")


@set_module("boost_histogram.axis")
class ArrayTuple(tuple):  # type: ignore
    __slots__ = ()
    # This is an exhaustive list as of NumPy 1.19
    _REDUCTIONS = {"sum", "any", "all", "min", "max", "prod"}

    def __getattr__(self, name: str) -> Any:
        if name in self._REDUCTIONS:
            return partial(getattr(np, name), np.broadcast_arrays(*self))
        else:
            return self.__class__(getattr(a, name) for a in self)

    def __dir__(self) -> List[str]:
        names = dir(self.__class__) + dir(np.ndarray)
        return sorted(n for n in names if not n.startswith("_"))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.__class__(a(*args, **kwargs) for a in self)

    def broadcast(self: A) -> A:
        """
        The arrays in this tuple will be compressed if possible to save memory.
        Use this method to broadcast them out into their full memory
        representation.
        """
        return self.__class__(np.broadcast_arrays(*self))


B = TypeVar("B", bound="AxesTuple")


@set_module("boost_histogram.axis")
class AxesTuple(tuple):  # type: ignore
    __slots__ = ()
    _MGRIDOPTS = {"sparse": True, "indexing": "ij"}

    @property
    def size(self) -> Tuple[int, ...]:
        return tuple(s.size for s in self)

    @property
    def extent(self) -> Tuple[int, ...]:
        return tuple(s.extent for s in self)

    @property
    def centers(self) -> ArrayTuple:
        gen = (s.centers for s in self)
        return ArrayTuple(np.meshgrid(*gen, **self._MGRIDOPTS))

    @property
    def edges(self) -> ArrayTuple:
        gen = (s.edges for s in self)
        return ArrayTuple(np.meshgrid(*gen, **self._MGRIDOPTS))

    @property
    def widths(self) -> ArrayTuple:
        gen = (s.widths for s in self)
        return ArrayTuple(np.meshgrid(*gen, **self._MGRIDOPTS))

    def value(self, *indexes: float) -> Tuple[float, ...]:
        if len(indexes) != len(self):
            raise IndexError(
                "Must have the same number of arguments as the number of axes"
            )
        return tuple(self[i].value(indexes[i]) for i in range(len(indexes)))

    def bin(self, *indexes: float) -> Tuple[float, ...]:
        if len(indexes) != len(self):
            raise IndexError(
                "Must have the same number of arguments as the number of axes"
            )
        return tuple(self[i].bin(indexes[i]) for i in range(len(indexes)))

    def index(self, *values: float) -> Tuple[float, ...]:  # type: ignore
        if len(values) != len(self):
            raise IndexError(
                "Must have the same number of arguments as the number of axes"
            )
        return tuple(self[i].index(values[i]) for i in range(len(values)))

    def __getitem__(self, item: Any) -> Any:
        result = super().__getitem__(item)
        return self.__class__(result) if isinstance(result, tuple) else result

    def __getattr__(self, attr: str) -> Any:
        return self.__class__(getattr(s, attr) for s in self)

    def __setattr__(self, attr: str, values: Any) -> None:
        self.__class__(s.__setattr__(attr, v) for s, v in zip(self, values))

    value.__doc__ = Axis.value.__doc__
    index.__doc__ = Axis.index.__doc__
    bin.__doc__ = Axis.bin.__doc__
