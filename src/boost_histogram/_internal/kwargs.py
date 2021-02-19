# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

from typing import Any, Dict, Set

del absolute_import, division, print_function


class KWArgs(object):
    def __init__(self, kwargs):
        # type: (Dict[str, Any]) -> None
        self.kwargs = kwargs

    def __enter__(self):
        # type: () -> KWArgs
        return self

    def __exit__(self, *args):
        # type: (Any) -> None
        if self.kwargs:
            raise TypeError("Keyword(s) {} not expected".format(", ".join(self.kwargs)))

    def __contains__(self, item):
        # type: (str) -> bool
        return item in self.kwargs

    def required(self, name):
        # type: (str) -> None
        if name in self.kwargs:
            self.kwargs.pop(name)
        else:
            raise KeyError("{} is required".format(name))

    def optional(self, name, default=None):
        # type: (str, Any) -> Any
        if name in self.kwargs:
            return self.kwargs.pop(name)
        else:
            return default

    def options(self, **options):
        # type: (bool) -> Set[str]
        return {option for option in options if self.optional(option, options[option])}
