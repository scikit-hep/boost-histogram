# -*- coding: utf-8 -*-
import inspect

import env
import pytest

import boost_histogram as bh
from boost_histogram._internal.sig_tools import make_signature_params

pytestmark = pytest.mark.skipif(env.PY2, reason="Python 2 does not have signatures")


def test_simple_sigs():
    from inspect import Parameter

    assert make_signature_params("x, y") == [
        Parameter("x", Parameter.POSITIONAL_OR_KEYWORD),
        Parameter("y", Parameter.POSITIONAL_OR_KEYWORD),
    ]

    assert make_signature_params("x=True") == [
        Parameter("x", Parameter.POSITIONAL_OR_KEYWORD, default=True)
    ]
    assert make_signature_params("*, x=None") == [
        Parameter("x", Parameter.KEYWORD_ONLY, default=None)
    ]
    assert make_signature_params("y : 'wow'") == [
        Parameter("y", Parameter.POSITIONAL_OR_KEYWORD, annotation="wow")
    ]
    assert make_signature_params("y : None = 2") == [
        Parameter("y", Parameter.POSITIONAL_OR_KEYWORD, default=2, annotation=None)
    ]
    assert make_signature_params("*args, **kwargs") == [
        Parameter("args", Parameter.VAR_POSITIONAL),
        Parameter("kwargs", Parameter.VAR_KEYWORD),
    ]

    assert make_signature_params("*args, x={b'x':3}") == [
        Parameter("args", Parameter.VAR_POSITIONAL),
        Parameter("x", Parameter.KEYWORD_ONLY, default={b"x": 3}),
    ]


def test_fill_sig():
    from inspect import Parameter

    a, b, c, d, e = inspect.signature(bh.Histogram.fill).parameters.values()

    assert a == Parameter("self", Parameter.POSITIONAL_OR_KEYWORD)
    assert b == Parameter("args", Parameter.VAR_POSITIONAL)
    assert c == Parameter("weight", Parameter.KEYWORD_ONLY, default=None)
    assert d == Parameter("sample", Parameter.KEYWORD_ONLY, default=None)
    assert e == Parameter("threads", Parameter.KEYWORD_ONLY, default=None)
