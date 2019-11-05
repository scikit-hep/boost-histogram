from __future__ import absolute_import, division, print_function

del absolute_import, division, print_function


def _walk_subclasses(cls):
    for base in cls.__subclasses__():
        # Find the furthest child to allow
        # user subclasses to work
        for inner in _walk_subclasses(base):
            yield inner
        yield base
