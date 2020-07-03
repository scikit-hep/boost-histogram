# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function


def _load():
    from .. import accumulators as acc
    import string

    tr = {ord(k): "_" + k.lower() for k in string.ascii_uppercase}

    # from CamelCase to snake_case
    r = {}
    for key in dir(acc):
        if key.startswith("_"):
            continue
        nkey = key[0].lower() + key[1:].translate(tr)
        r[nkey] = getattr(acc, key)
    return r


locals().update(_load())

del absolute_import, division, print_function
del _load

# These will have the original module locations and original names.
