import inspect
from typing import Any, Callable, Dict, Optional


def make_signature_params(sig, locals=None):
    # type: (str, Optional[Dict[str, Any]]) -> Any
    if locals is None:
        locals = {}
    exec(f"def _({sig}): pass", globals(), locals)
    return list(inspect.signature(locals["_"]).parameters.values())


def inject_signature(sig, locals=None):
    # type: (str, Optional[Dict[str, Any]]) -> Any
    if locals is None:
        locals = {}

    def wrap(f):
        # type: (Callable[..., Any]) -> Callable[..., Any]
        # It is invalid to have a positional only argument till Python 3.8
        # We could split on / as well

        params = make_signature_params(sig, locals)

        signature = inspect.signature(f)
        signature = signature.replace(parameters=params)
        f.__signature__ = signature  # type: ignore
        return f

    return wrap
