# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

from typing import Tuple, Type

del absolute_import, division, print_function

# This is a Python 2-3 compat file. Anything added here must exactly match the
# six package.

# This will be (unicode, str) in Python 2, and (str,) in Python 3
string_types = tuple({type(u""), str})  # type: Tuple[Type, ...]
