from .core.axis import regular_log, regular_sqrt, regular_pow, circular, options

from .core import axis as ca

from .utils import FactoryMeta, KWArgs


# When Python 2 is dropped, this could use keyword
# only argument syntax instead of kwargs
def _make_regular(bins, start, stop, **kwargs):
    """
    Make a regular axis with nice keyword arguments for underflow,
    overflow, and growth.
    """
    with KWArgs(kwargs) as k:
        metadata = k.optional("metadata")
        options = k.options(underflow=True, overflow=True, growth=False, circular=False)

    if options == {"growth", "underflow", "overflow"}:
        return ca._regular_uoflow_growth(bins, start, stop, metadata)
    elif options == {"underflow", "overflow"}:
        return ca._regular_uoflow(bins, start, stop, metadata)
    elif options == {"underflow"}:
        return ca._regular_uflow(bins, start, stop, metadata)
    elif options == {"overflow"}:
        return ca._regular_oflow(bins, start, stop, metadata)
    elif options == {"circular", "underflow", "overflow"}:
        return circular(bins, start, stop, metadata)
    elif options == set():
        return ca._regular_none(bins, start, stop, metadata)
    else:
        raise KeyError("Unsupported collection of options")


regular = FactoryMeta(
    _make_regular,
    (
        ca._regular_none,
        ca._regular_uflow,
        ca._regular_oflow,
        ca._regular_uoflow,
        ca._regular_uoflow_growth,
        regular_log,
        regular_sqrt,
        regular_pow,
        circular,
    ),
)


def _make_variable(edges, **kwargs):
    """
    Make a variable axis with nice keyword arguments for underflow,
    overflow, and growth.
    """
    with KWArgs(kwargs) as k:
        metadata = k.optional("metadata")
        options = k.options(underflow=True, overflow=True, growth=False)

    if options == {"growth", "underflow", "overflow"}:
        return ca._variable_uoflow_growth(edges, metadata)
    elif options == {"underflow", "overflow"}:
        return ca._variable_uoflow(edges, metadata)
    elif options == {"underflow"}:
        return ca._variable_uflow(edges, metadata)
    elif options == {"overflow"}:
        return ca._variable_oflow(edges, metadata)
    elif options == set():
        return ca._variable_none(edges, metadata)
    else:
        raise KeyError("Unsupported collection of options")


variable = FactoryMeta(
    _make_variable,
    (
        ca._variable_none,
        ca._variable_uflow,
        ca._variable_oflow,
        ca._variable_uoflow,
        ca._variable_uoflow_growth,
    ),
)


def _make_integer(start, stop, **kwargs):
    """
    Make an integer axis with nice keyword arguments for underflow,
    overflow, and growth.
    """
    with KWArgs(kwargs) as k:
        metadata = k.optional("metadata")
        options = k.options(underflow=True, overflow=True, growth=False)

    # underflow and overflow settings are ignored, integers are always
    # finite and thus cannot end up in a flow bin when growth is on
    if "growth" in options and "circular" not in options:
        return ca._integer_growth(start, stop, metadata)
    elif options == {"underflow", "overflow"}:
        return ca._integer_uoflow(start, stop, metadata)
    elif options == {"underflow"}:
        return ca._integer_uflow(start, stop, metadata)
    elif options == {"overflow"}:
        return ca._integer_oflow(start, stop, metadata)
    elif options == set():
        return ca._integer_none(start, stop, metadata)
    else:
        raise KeyError("Unsupported collection of options")


integer = FactoryMeta(
    _make_integer,
    (
        ca._integer_none,
        ca._integer_uflow,
        ca._integer_oflow,
        ca._integer_uoflow,
        ca._integer_growth,
    ),
)


def _make_category(categories, **kwargs):
    """
    Make a category axis with ints or strings and with nice keyword
    arguments for growth.
    """
    with KWArgs(kwargs) as k:
        metadata = k.optional("metadata")
        options = k.options(growth=False)

    if isinstance(categories, str):
        categories = list(categories)

    if options == {"growth"}:
        try:
            return ca._category_int_growth(categories, metadata)
        except TypeError:
            return ca._category_str_growth(categories, metadata)
    elif options == set():
        try:
            return ca._category_int(categories, metadata)
        except TypeError:
            return ca._category_str(categories, metadata)
    else:
        raise KeyError("Unsupported collection of options")


category = FactoryMeta(
    _make_category,
    (
        ca._category_int,
        ca._category_int_growth,
        ca._category_str,
        ca._category_str_growth,
    ),
)

del FactoryMeta
