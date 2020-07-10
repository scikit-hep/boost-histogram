# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

del absolute_import, division, print_function


class KWArgs(object):
    def __init__(self, kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.kwargs:
            raise TypeError("Keyword(s) {} not expected".format(", ".join(self.kwargs)))

    def __contains__(self, item):
        return item in self.kwargs

    def required(self, name):
        if name in self.kwargs:
            self.kwargs.pop(name)
        else:
            raise KeyError("{0} is required".format(name))

    def optional(self, name, default=None):
        if name in self.kwargs:
            return self.kwargs.pop(name)
        else:
            return default

    def options(self, **options):
        return {option for option in options if self.optional(option, options[option])}
