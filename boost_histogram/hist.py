from .utils import FactoryMeta, KWArgs

from . import core as _core

_histogram_and_storage = {
    _core.storage.double: _core.hist._any_double,
    _core.storage.int: _core.hist._any_int,
    _core.storage.atomic_int: _core.hist._any_atomic_int,
    _core.storage.unlimited: _core.hist._any_unlimited,
    _core.storage.weight: _core.hist._any_weight,
    _core.storage.mean: _core.hist._any_mean,
    _core.storage.weighted_mean: _core.hist._any_weighted_mean,
}


def _arg_shortcut(item):
    if isinstance(item, tuple):
        return _core.axis._regular_uoflow(*item)
    else:
        return item


def _make_histogram(*args, **kwargs):
    """
    Make a histogram with an optional storage (keyword only).
    """

    with KWArgs(kwargs) as k:
        storage = k.optional("storage", _core.storage.double())

    # Initialize storage if user has not
    if isinstance(storage, type):
        storage = storage()

    args = [_arg_shortcut(arg) for arg in args]

    for s in _histogram_and_storage:
        if isinstance(storage, s):
            return _histogram_and_storage[s](args, storage)

    raise TypeError("Unsupported storage")


histogram = FactoryMeta(_make_histogram, tuple(_histogram_and_storage.values()))
