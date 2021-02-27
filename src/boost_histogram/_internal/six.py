from typing import Tuple, Type

# This is a Python 2-3 compat file. Anything added here must exactly match the
# six package.

# This will be (unicode, str) in Python 2, and (str,) in Python 3
string_types = tuple({type(""), str})  # type: Tuple[Type, ...]
