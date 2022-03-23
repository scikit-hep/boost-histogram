.. _usage-subclassing:


Subclassing (advanced)
======================

Subclassing boost-histogram components is supported, but requires a little extra care to ensure the subclasses do not return un-wrapped boost-histogram components when a subclassed version is available. The issue is that various actions make the C++ -> Python transition over again, such as using ``.project()``. For example, let's say you have a ``MyHistogram`` and a ``MyRegular``. If you use ``project(0)``, that needs to also return a ``MyRegular``, but it is reconverting the return value from C++ to Python, so it has to somehow know that ``MyRegular`` is the right axis subclass to select from for ``MyHistogram``. This is accomplished with families.

When you subclass, you will need to add a family. Any object can be used - the module for your library is a good choice if you only have one "family" of histograms. Boost-histogram uses ``boost_histogram``, Hist uses ``hist``. You can use anything you want, though; a custom tag object like ``MY_FAMILY = object()`` works well too. It just has to support ``is``, and be the exact same object on all your subclasses.

.. code:: python3

    import boost_histogram as bh
    import my_package

    class Histogram(bh.Histogram, family=my_package):
        ...

    class Regular(bh.axis.Regular, family=my_package):
        ...

If you only override ``Histogram``, you can leave off the ``family=`` argument, or set it to ``None``. It will generate a private ``object()`` in this case. You must add an explicit family to ``Histogram`` if you subclass any further components.

If you use Mixins, special care needs to be taken if you need a left-acting
Mixin, since class keywords are handled via ``super()`` left to right. This is
a Mixin that will work on either side:

.. code:: python3

    class AxisMixin:
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)  # type: ignore

Mixins are recommended if you want to provide functionality to a collection of
different subclasses, like ``Axis``.

There are customization hooks provided for subclasses as well.
``self._generate_axis_()`` is called to produce an ``AxesTuple``, so you can
override that if you customize ``AxesTuple``. ``_import_bh_`` and
``_export_bh_`` are called when converting an object between histogram
libraries. ``cls._export_bh(self)`` is called from the outgoing class (being
converted from), and ``self._import_bh_()`` is called afterward on the incoming
class (being converted to). So if ``h1`` is an instance of ``H1``, and ``H2``
is the new class, then ``H2(h1)`` calls ``H1._export_bh_(h2)`` and then
``h2._import_bh()`` before returning ``h2``. The internal repr building for axes is
a list produced by ``_repr_args_`` representing each item in the repr.
