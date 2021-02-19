# -*- coding: utf-8 -*-
try:
    from cPickle import pickle
except ImportError:
    import pickle

import os

import env
import pytest

import boost_histogram as bh

DIR = os.path.abspath(os.path.dirname(__file__))


@pytest.mark.xfail(
    env.PY2 and env.CPYTHON,
    raises=TypeError,
    reason="Python 3 pickle can't be read in CPython 2",
)
@pytest.mark.parametrize("version", ["0.10.2", "0.6.2", "0.11.1"])
def test_read_pickle(version):

    filename = os.path.join(DIR, "pickles", "bh_{}.pkl".format(version))
    with open(filename, "rb") as f:
        d = pickle.load(f)

    assert d["version"] == version

    h1 = d["h1"]
    h2 = d["h2"]
    h3 = d["h3"]

    assert h1._storage_type == bh.storage.Double
    assert h2._storage_type == bh.storage.Weight
    assert h3._storage_type == bh.storage.Double

    assert h1[bh.loc(-5)] == 1
    assert h1[bh.loc(1)] == 2
    assert h1[bh.loc(2)] == 1

    assert h2[0].value == 0
    assert h2[1].value == 1
    assert h2[2].value == 2
    assert h2[3].value == 3
    assert h2[4].value == 0

    assert h3[bh.loc("one"), bh.loc(3)] == 1
    assert h3[bh.loc("two"), bh.loc(2)] == 1
    assert h3[bh.loc("two"), bh.loc(1)] == 1
    assert h3[bh.loc("two"), bh.loc(3)] == 0
    assert h3[bh.loc("two"), sum] == 2

    assert isinstance(h1.axes[0], bh.axis.Regular)
    assert isinstance(h2.axes[0], bh.axis.Integer)
    assert isinstance(h3.axes[0], bh.axis.StrCategory)
    assert isinstance(h3.axes[1], bh.axis.Variable)

    assert h3.axes[0].traits.growth
    assert not h3.axes[1].traits.growth

    assert h1.axes[0].metadata is None
    assert h2.axes[0].metadata == {"hello": "world"}

    ver = tuple(map(int, version.split(".")))

    assert h1.metadata is None

    if ver < (0, 9, 0):
        assert h2.metadata is None
    else:
        assert h2.metadata == "foo"
