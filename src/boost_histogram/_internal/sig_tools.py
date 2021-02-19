# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import inspect
import sys
from typing import Any, Callable, Dict, Optional

del absolute_import, division, print_function


if sys.version_info < (3, 0):
    make_signature_params = None

    def inject_signature(sig, locals=None):
        # type: (str, Optional[Dict[str, Any]]) -> Any
        def wrap(f):
            return f

        return wrap


else:

    def make_signature_params(sig, locals=None):
        # type: (str, Optional[Dict[str, Any]]) -> Any
        if locals is None:
            locals = {}
        exec("def _({}): pass".format(sig), globals(), locals)
        return list(inspect.signature(locals["_"]).parameters.values())

    def inject_signature(sig, locals=None):
        # type: (str, Optional[Dict[str, Any]]) -> Any
        if locals is None:
            locals = {}

        def wrap(f):
            # type: (Callable[..., Any]) -> Callable[..., Any]
            # It is invalid to have a positional only argument till Python 3.8
            # We could split on / as well

            params = make_signature_params(sig, locals)

            signature = inspect.signature(f)
            signature = signature.replace(parameters=params)
            f.__signature__ = signature  # type: ignore
            return f

        return wrap
