# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import inspect

del absolute_import, division, print_function


def make_signature_params(sig, locals=None):
    if locals is None:
        locals = {}
    exec("def _({0}): pass".format(sig), globals(), locals)
    return list(inspect.signature(locals["_"]).parameters.values())


def inject_signature(sig, locals=None):
    if locals is None:
        locals = {}

    def wrap(f):
        # Don't add on Python 2
        if not hasattr(inspect, "Parameter"):
            return f

        # It is invalid to have a positonal only argument till Python 3.8
        # We could split on / as well

        params = make_signature_params(sig, locals)

        signature = inspect.signature(f)
        signature = signature.replace(parameters=params)
        f.__signature__ = signature
        return f

    return wrap
