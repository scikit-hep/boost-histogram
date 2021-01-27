# -*- coding: utf-8 -*-
# The point of this test is to make sure that the infrastructure for supporting
# custom attributes, like title in Hist, is working.

import pytest

import boost_histogram as bh

# First, make a new family to identify your library
CUSTOM_FAMILY = object()


# Add named axes
class NamedAxesTuple(bh.axis.AxesTuple):
    __slots__ = ()

    def _get_index_by_name(self, name):
        if isinstance(name, str):
            for i, ax in enumerate(self):
                if ax.name == name:
                    return i
            raise KeyError("{} not found in axes".format(name))
        else:
            return name

    def __getitem__(self, item):
        if isinstance(item, slice):
            item = slice(
                self._get_index_by_name(item.start),
                self._get_index_by_name(item.stop),
                self._get_index_by_name(item.step),
            )
        else:
            item = self._get_index_by_name(item)

        return super(NamedAxesTuple, self).__getitem__(item)

    @property
    def name(self):
        """
        The names of the axes. May be empty strings.
        """
        return tuple(ax.name for ax in self)


# When you subclass Histogram or an Axes, you should register your family so
# boost-histogram will know what to convert C++ objects into.


class AxesMixin(object):
    __slots__ = ()

    @property
    def name(self):
        """
        Get the name for the Regular axis
        """
        return self._ax.metadata.get("name", "")


# The order of the mixin is important here - it must be first
# to override bh.axis.Regular
@bh.utils.set_family(CUSTOM_FAMILY)
class Regular(bh.axis.Regular, AxesMixin):
    __slots__ = ()

    def __init__(self, bins, start, stop, name):

        super(Regular, self).__init__(bins, start, stop)

        self._ax.metadata["name"] = name


@bh.utils.set_family(CUSTOM_FAMILY)
class CustomHist(bh.Histogram):
    def _generate_axes_(self):
        return NamedAxesTuple(self._axis(i) for i in range(self.ndim))

    def __init__(self, *args, **kwargs):
        super(CustomHist, self).__init__(*args, **kwargs)
        valid_names = [ax.name for ax in self.axes if ax.name]
        if len(valid_names) != len(set(valid_names)):
            msg = "{} instance cannot contain axes with duplicated names".format(
                self.__class__.__name__
            )
            raise KeyError(msg)


def test_hist_creation():
    hist_1 = CustomHist(Regular(10, 0, 1, name="a"), Regular(20, 0, 4, name="b"))
    assert hist_1.axes[0].name == "a"
    assert hist_1.axes[1].name == "b"

    hist_2 = CustomHist(Regular(10, 0, 1, name=""), Regular(20, 0, 4, name=""))
    assert hist_2.axes[0].name == ""
    assert hist_2.axes[1].name == ""

    with pytest.raises(KeyError):
        CustomHist(Regular(10, 0, 1, name="a"), Regular(20, 0, 4, name="a"))


def test_hist_index():
    hist_1 = CustomHist(Regular(10, 0, 1, name="a"), Regular(20, 0, 4, name="b"))
    assert hist_1.axes[0].name == "a"
    assert hist_1.axes[1].name == "b"


def test_hist_convert():
    hist_1 = CustomHist(Regular(10, 0, 1, name="a"), Regular(20, 0, 4, name="b"))
    hist_bh = bh.Histogram(hist_1)

    assert type(hist_bh.axes[0]) == bh.axis.Regular
    assert hist_bh.axes[0].name == "a"
    assert hist_bh.axes[1].name == "b"

    hist_2 = CustomHist(hist_bh)

    assert type(hist_2.axes[0]) == Regular
    assert hist_2.axes[0].name == "a"
    assert hist_2.axes[1].name == "b"

    # Just verify no-op status
    hist_3 = CustomHist(hist_1)

    assert type(hist_3.axes[0]) == Regular
    assert hist_3.axes[0].name == "a"
    assert hist_3.axes[1].name == "b"


def test_access():
    hist = CustomHist(Regular(10, 0, 1, name="a"), Regular(20, 0, 4, name="b"))

    assert hist.axes["a"] == hist.axes[0]
    assert hist.axes["b"] == hist.axes[1]

    from_bh = bh.Histogram(bh.axis.Regular(10, 0, 1), bh.axis.Regular(20, 0, 4))
    from_bh.axes.name = "a", "b"
    hist_conv = CustomHist(from_bh)

    assert hist_conv.axes["a"] == hist_conv.axes[0]
    assert hist_conv.axes["b"] == hist_conv.axes[1]
