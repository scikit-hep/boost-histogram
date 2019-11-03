from __future__ import absolute_import, division, print_function

del absolute_import, division, print_function

import inspect
import itertools
import ast

# Python 2 filter (function defs need to be Python 2 compatible)
EMPTY = inspect.Parameter.empty if hasattr(inspect, "Parameter") else None


def make_param(arg, kind, default=EMPTY):
    empty = lambda x: inspect.Parameter.empty if x is None else x[0]
    return inspect.Parameter(
        arg.arg, kind, annotation=empty(arg.annotation), default=default
    )


def run_on(args, defaults, kind):
    params = []
    arg_def = reversed(
        list(
            itertools.zip_longest(
                reversed(args), reversed(defaults), fillvalue=inspect.Parameter.empty
            )
        )
    )

    return [make_param(arg, kind, default) for arg, default in arg_def]


class ArgTrans(ast.NodeTransformer):
    def __init__(self, locals={}):
        self.locals = locals

    def visit_Tuple(self, node):
        node = self.generic_visit(node)
        return (tuple(node.elts),)

    def visit_Set(self, node):
        node = self.generic_visit(node)
        return (set(node.elts),)

    def visit_Dict(self, node):
        node = self.generic_visit(node)
        return (dict(zip(node.keys, node.values)),)

    def visit_Name(self, node):
        return (eval(node.id, globals(), self.locals),)

    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        return (getattr(node.value[0], node.attr),)

    def visit_Call(self, node):
        "Simplified - can only do f(), no args"
        node = self.generic_visit(node)
        return (node.func[0](*node.args),)

    # Python 3.8+
    def visit_Constant(self, node):
        return (node.value,)

    # Python < 3.8
    def visit_NameConstant(self, node):
        return (node.value,)

    # Python < 3.8
    def visit_Str(self, node):
        return (node.s,)

        # Python < 3.8

    def visit_Bytes(self, node):
        return (node.s,)

    # Python < 3.8
    def visit_Num(self, node):
        return (node.n,)

    def visit_Ellipsis(self, node):
        return (Ellipsis,)

    def visit_arguments(self, node):
        node = self.generic_visit(node)
        empty = lambda x: inspect.Parameter.empty if x is None else x
        params = []

        params += run_on(
            node.args, node.defaults, inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
        if node.vararg:
            params.append(make_param(node.vararg, inspect.Parameter.VAR_POSITIONAL))
        params += run_on(
            node.kwonlyargs, node.kw_defaults, inspect.Parameter.KEYWORD_ONLY
        )
        if node.kwarg:
            params.append(make_param(node.kwarg, inspect.Parameter.VAR_KEYWORD))

        return params


def make_signature_params(sig, locals={}):
    s = ast.parse("def f({0}): pass".format(sig))
    return ArgTrans(locals).visit(s.body[0].args)


def inject_signature(sig, locals={}):
    def wrap(f):
        # Don't add on Python 2
        if not hasattr(inspect, "Parameter"):
            return f

        # It is invalid to have a positonal only argument till Python 3.8
        # If we avoided the ast, we could do it earlier here
        # We could split on / as well

        params = make_signature_params(sig, locals)

        signature = inspect.signature(f)
        signature = signature.replace(parameters=params)
        f.__signature__ = signature
        return f

    return wrap
