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

.. py:function:: regular(bins, start, stop, metadata="", underflow=True, overflow=True, grow=False)

   This also takes ``flow=False`` as a quick way to disable both underflow and overflow.

There are some other useful axis types based on regular axis:


.. image:: ../_images/axis_circular.png
   :alt: Regular axis illustration
   :align: center

.. py:function:: circular(bins, start, stop, metadata="")

   This wraps around.

.. py:function:: regular_sqrt(bins, start, stop, metadata="")

   Transformed by a sqrt.

.. py:function:: regular_log(bins, start, stop, metadata="")

   Transformed by log.

.. py:function:: regular_pow(power, bins, start, stop, metadata="")

   Transformed by a power (first argument sets the power).

Variable axis
-------------

.. image:: ../_images/axis_variable.png
   :alt: Regular axis illustration
   :align: center

.. py:function:: variable([edge1, ...], metadata="", underflow=True, overflow=True)

   You can set the bin edges explicitly.

Integer axis
------------

.. image:: ../_images/axis_integer.png
   :alt: Regular axis illustration
   :align: center

.. py:function:: integer(start, stop, metadata="", underflow=True, overflow=True, grow=False)

   This could be mimicked with a regular axis, but is simpler and slightly faster.

Category axis
-------------

.. image:: ../_images/axis_category.png
   :alt: Regular axis illustration
   :align: center

.. py:function:: category([value1, ...], metadata="", grow=False)

   You should put integers in a category axis; but unlike an integer axis, the integers do not need to be adjacent.
