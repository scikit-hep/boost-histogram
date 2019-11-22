Axes
====

In boost-histogram, a histogram is collection of Axis objects and a
storage.

There are several axis types to choose from.

Regular axis
------------

.. image:: ../_images/axis_regular.png
   :alt: Regular axis illustration
   :align: center

.. py:function:: bh.axis.Regular(bins, start, stop, *, metadata="", underflow=True, overflow=True, circular=False, grow=False, transform=None)

   This also takes ``flow=False`` as a quick way to disable both underflow and overflow.

There are some other useful axis types based on regular axis:


.. image:: ../_images/axis_circular.png
   :alt: Regular axis illustration
   :align: center

.. py:function:: bh.axis.Regular(..., circular=True)

   This wraps around.

.. py:function:: bh.axis.Regular(..., transform=bh.axis.transform.sqrt)

   Transformed by a sqrt.

.. py:function:: bh.axis.Regular(..., transform=bh.axis.transform.log)

   Transformed by log.

.. py:function:: bh.axis.Regular(..., transform=bh.axis.transform.Power(v))

   Transformed by a power (the argument is the power).

There is also a system for adding your own transform functions; see :ref:`usage-transforms`.

Variable axis
-------------

.. image:: ../_images/axis_variable.png
   :alt: Regular axis illustration
   :align: center

.. py:function:: bh.axis.Variable([edge1, ...], *, metadata="", underflow=True, overflow=True, circular=False, growth=False)

   You can set the bin edges explicitly.

Integer axis
------------

.. image:: ../_images/axis_integer.png
   :alt: Regular axis illustration
   :align: center

.. py:function:: bh.axis.Integer(start, stop, *, metadata="", underflow=True, overflow=True, grow=False)

   This could be mimicked with a regular axis, but is simpler and slightly faster.

Category axis
-------------

.. image:: ../_images/axis_category.png
   :alt: Regular axis illustration
   :align: center

.. py:function:: bh.axis.IntCategory([value1, ...], metadata="", grow=False)

   You should put integers in a category axis; but unlike an integer axis, the integers do not need to be adjacent.

.. py:function:: bh.axis.StrCategory([str1, ...], metadata="", grow=False)

   You can put strings in a category axis as well.
