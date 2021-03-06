import functools
import warnings
from typing import Any

# Warning: this should not be directly used on properties. It will trigger on
# tab completion - ALL tab completion that could include this property.
# ob.<tab> will produce a warning, for example. Instead use a hidden method and
# a __getattr__ if the property was not settable.


class deprecated:
    def __init__(self, reason: str, name: str = "") -> None:
        self._reason = reason
        self._name = name

    def __call__(self, func: Any) -> Any:
        @functools.wraps(func)
        def decorated_func(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(
                "{} is deprecated: {}".format(
                    self._name or func.__name__, self._reason
                ),
                category=FutureWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        decorated_func.__doc__ = "DEPRECATED: " + self._reason + "\n" + func.__doc__
        return decorated_func
