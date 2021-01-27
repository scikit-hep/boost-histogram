# -*- coding: utf-8 -*-
"""
Using the protocol:

Producers: use isinstance(myhist, PlottableHistogram) in your tests; part of
the protocol is checkable at runtime, though ideally you should use MyPy; if
your histogram class supports PlottableHistogram, this will pass.

Consumers: Make your functions accept the PlottableHistogram static type, and
MyPy will force you to only use items in the Protocol.
"""

import sys
from typing import Any, Iterable, Optional, Sequence, Tuple, TypeVar, Union

if sys.version_info < (3, 8):
    from typing_extensions import Protocol, runtime_checkable

else:
    from typing import Protocol, runtime_checkable


protocol_version = 1

# from numpy.typing import ArrayLike # requires unreleased NumPy 1.20
ArrayLike = Iterable[float]

# Known kinds of histograms. A Producer can add Kinds not defined here; a
# Consumer should check for known types if it matters. A simple plotter could
# just use .value and .variance if non-None and ignore .kind.
#
# Could have been Kind = Literal["COUNT", "MEAN"] - left as a generic string so
# it can be extendable.
Kind = str

# Implementations are highly encouraged to use the following construct:
# class Kind(str, enum.Enum):
#     COUNT = "COUNT"
#     MEAN = "MEAN"
# Then return and use Kind.COUNT or Kind.MEAN.


@runtime_checkable
class PlottableTraits(Protocol):
    # True if the axis "wraps around"
    circular: bool

    # True if each bin is discrete - Integer, Boolean, or Category, for example
    discrete: bool


T = TypeVar("T", covariant=True)


@runtime_checkable
class PlottableAxisGeneric(Protocol[T]):
    # name: str - Optional, not part of Protocol
    # label: str - Optional, not part of Protocol
    #
    # Plotters are encouraged to plot label if it exists and is not None, and
    # name otherwise if it exists and is not None, but these properties are not
    # available on all histograms and not part of the Protocol.

    traits: PlottableTraits

    def __getitem__(self, index: int) -> T:
        """
        Get the pair of edges (not discrete) or bin label (discrete).
        """

    def __len__(self) -> int:
        """
        Return the number of bins (not counting flow bins, which are ignored
        for this Protocol currently).
        """

    def __eq__(self, other: Any) -> bool:
        """
        Required to be sequence-like.
        """


PlottableAxisContinuous = PlottableAxisGeneric[Tuple[float, float]]
PlottableAxisInt = PlottableAxisGeneric[int]
PlottableAxisStr = PlottableAxisGeneric[str]

PlottableAxis = Union[PlottableAxisContinuous, PlottableAxisInt, PlottableAxisStr]


@runtime_checkable
class PlottableHistogram(Protocol):
    axes: Sequence[PlottableAxis]

    kind: Kind

    # All methods can have a flow=False argument - not part of this Protocol.
    # If this is included, it should return an array with flow bins added,
    # normal ordering.

    def values(self) -> ArrayLike:
        """
        Returns the accumulated values. The counts for simple histograms, the
        sum of weights for weighted histograms, the mean for profiles, etc.

        If counts is equal to 0, the value in that cell is undefined if
        kind == "MEAN".
        """

    def variances(self) -> Optional[ArrayLike]:
        """
        Returns the estimated variance of the accumulated values. The sum of squared
        weights for weighted histograms, the variance of samples for profiles, etc.
        For an unweighed histogram where kind == "COUNT", this should return the same
        as values if the histogram was not filled with weights, and None otherwise.

        If counts is equal to 1 or less, the variance in that cell is undefined if
        kind == "MEAN".

        If kind == "MEAN", the counts can be used to compute the error on the mean
        as sqrt(variances / counts), this works whether or not the entries are
        weighted if the weight variance was tracked by the implementation.
        """

    def counts(self) -> Optional[ArrayLike]:
        """
        Returns the number of entries in each bin for an unweighted
        histogram or profile and an effective number of entries (defined below)
        for a weighted histogram or profile. An exotic generalized histogram could
        have no sensible .counts, so this is Optional and should be checked by
        Consumers.

        If kind == "MEAN", counts (effective or not) can and should be used to
        determine whether the mean value and its variance should be displayed
        (see documentation of values and variances, respectively). The counts
        should also be used to compute the error on the mean (see documentation
        of variances).

        For a weighted histogram, counts is defined as sum_of_weights ** 2 /
        sum_of_weights_squared. It is equal or less than the number of times
        the bin was filled, the equality holds when all filled weights are equal.
        The larger the spread in weights, the smaller it is, but it is always 0
        if filled 0 times, and 1 if filled once, and more than 1 otherwise.

        A suggested implementation is:

            return np.divide(
                sum_of_weights**2,
                sum_of_weights_squared,
                out=np.zeros_like(sum_of_weights, dtype=np.float64),
                where=sum_of_weights_squared != 0)
        """
