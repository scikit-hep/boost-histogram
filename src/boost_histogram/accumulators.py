# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function


def _load():
    from ._core import accumulators as acc

    r = {}
    for key in dir(acc):
        if key.startswith("_"):
            continue
        cls = getattr(acc, key)
        cls.__module__ = "boost_histogram.accumulators"
        r[key] = cls
    return r


locals().update(_load())
del absolute_import, division, print_function
del _load

# Not supported by PyBind builtins
# Enable if wrapper added
# inject_signature("self, value")(Sum.fill)
# inject_signature("self, value, *, variance=None")(WeightedSum.fill)
# inject_signature("self, value, *, weight=None")(Mean.fill)
# inject_signature("self, value, *, weight=None")(WeightedMean.fill)
