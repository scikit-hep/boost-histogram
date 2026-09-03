from __future__ import annotations

from boost_histogram._utils import register


def test_register_subclass_does_not_mutate_parent_types():
    # register() must give each decorated class its own "_types" set - it
    # used to inherit the parent's set via hasattr() and mutate it in place.
    class CppBase:
        pass

    class CppChild(CppBase):
        pass

    @register({CppBase})
    class Parent:
        pass

    @register({CppChild})
    class Child(Parent):
        pass

    assert Parent._types == {CppBase}
    assert Child._types == {CppChild}
