from .core.axis import regular_log, regular_sqrt, regular_pow, circular, options

from .core import axis as ca

from .utils import FactoryMeta

regular = FactoryMeta(
    ca._make_regular,
    (
        ca._regular_uoflow,
        ca._regular_uflow,
        ca._regular_oflow,
        ca._regular_noflow,
        ca._regular_growth,
    ),
)

variable = FactoryMeta(
    ca._make_variable,
    (ca._variable_uoflow, ca._variable_uflow, ca._variable_oflow, ca._variable_noflow),
)

integer = FactoryMeta(
    ca._make_integer,
    (
        ca._integer_uoflow,
        ca._integer_uflow,
        ca._integer_oflow,
        ca._integer_noflow,
        ca._integer_growth,
    ),
)

category = FactoryMeta(
    ca._make_category,
    (
        ca._category_int,
        ca._category_int_growth,
        ca._category_str,
        ca._category_str_growth,
    ),
)

del ca, FactoryMeta
