.. _usage-transforms:

Using Transforms
================

The boost-histogram library provides a powerful transform system on Regular axes that allows
you to provide a functional form for the conversion between a regular spacing and the actual
bin edges. The following transforms are built in:


* ``bh.axis.transform.sqrt``: A square root transform
* ``bh.axis.transform.log``: A logarithmic transform
* ``bh.axis.transform.Pow(power)`` Raise to a specified power (``power=0.5`` is identical to ``sqrt``)

There is also a flexible ``bh.axis.transform.Function``, which allows you to specify arbitrary conversion functions (detailed below).


Simple custom transforms
------------------------


The ``Function`` transform takes two ctypes ``double(double)`` function pointers, a forward transform and a inverse transform. An object that provides a ctypes function pointer through a ``.ctypes`` attribute is supported, as well. As an example, let's look at how one would recreate the ``log`` transform using several different methods:

Pure Python
^^^^^^^^^^^

You can directly cast a python callable to a ctypes pointer, and use that. However, you will call Python *every* time you interact with the
transformed axis, and this will be 15-90 times slower than a compiled method, like ``bh.axis.transform.log``. In most cases, a Variable axis will be faster.

.. code-block:: python

   import ctypes
   ftype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)

   # Pure Python (15x slower)
   bh.axis.Regular(10, 1, 4, transform=bh.axis.transform.Function(ftype(math.log), ftype(math.exp)))

   # Pure Python: Numpy (90x slower)
   bh.axis.Regular(10, 1, 4, transform=bh.axis.transform.Function(ftype(np.log), ftype(np.exp)))

You can create a Variable axis from the edges of this axis; often that will be faster.

You can also use ``transform=ftype`` and just directly provide the functions; this provides nicer
reprs, but is still not picklable because ftype is a generated and not picklable; see below
for a way to make this picklable. You can also specify ``name="..."`` to customize the repr explicitly.

Using Numba
^^^^^^^^^^^

If you have the numba library installed, and your transform is reasonably simple, you can use the ``@numba.cfunc`` decorator to create
a callable that will run directly through the C interface. This is just as fast as the compiled version provided!

.. code-block:: python

   import numba

   @numba.cfunc(numba.float64(numba.float64))
   def exp(x):
       return math.exp(x)

   @numba.cfunc(numba.float64(numba.float64))
   def log(x):
       return math.log(x)

   bh.axis.Regular(10, 1, 4, transform=bh.axis.transform.Function(log, exp))

Manual compilation
^^^^^^^^^^^^^^^^^^

You can also get a ctypes pointer from the usual place: a library. Let's say you have the following ``mylib.c`` file:

.. code-block:: c

   #include <math.h>

   double my_log(double value) {
       return log(value);
   }

   double my_exp(double value) {
       return exp(value);
   }


And you compile it with:

.. code-block:: bash

   gcc mylib.c -shared -o mylib.so

You can now use it like this:

.. code-block:: python

   import ctypes
   ftype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)

   mylib = ctypes.CDLL("mylib.so")

   my_log = ctypes.cast(mylib.my_log, ftype)
   my_exp = ctypes.cast(mylib.my_exp, ftype)

   bh.axis.Regular(10, 1, 4, transform=bh.axis.transform.Function(my_log, my_exp))


Note that you do actually have to cast it to the correct function type; just setting
``argtypes`` and ``restype`` does not work.

Picklable custom transforms
---------------------------

The above examples to not support pickling, since ctypes pointers (or pointers in general)
are not picklable. However, the ``Function`` transform supports a ``convert=`` keyword
argument that takes the two provided objects and converts them to ctypes pointers.
So if you can supply a pair of picklable objects and a conversion function, you can
make a fully picklable transform. A few common cases are given below.

Pure Python
^^^^^^^^^^^

This is the easiest example; as long as your Python function is picklable, all you need to do is move the
ctypes call into the convert function. You need a little wrapper function to make it picklable:

.. code-block:: python

   import ctypes, math

   # We need a little wrapper function only because `ftype` is not directly picklable
   def convert_python(func):
       ftype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
       return ftype(func)

   bh.axis.Regular(10, 1, 4, transform=bh.axis.transform.Function(math.log, math.exp, convert=convert_python))

That's it.

Using Numba
^^^^^^^^^^^

The same procedure works for numba decorators. Numpy only supports functions, not builtins like ``math.log``,
so if you want to pass those, you'll need to wrap them in a lambda function or add a bit of logic to the convert
function. Here are your options:

.. code-block:: python

    import numba, math

    def convert_numba(func):
        return numba.cfunc(numba.double(numba.double))(func)

    # Built-ins and ufuncs need to be wrapped (numba can't read a signature)
    # User functions would not need the lambda
    bh.axis.Regular(10, 1, 4,
                    transform=bh.axis.transform.Function(lambda x: math.log(x), lambda x: math.exp(x),
                                                         convert=convert_numba))

Note that ``numba.cfunc`` does not work on its own builtins, but requires a user function. Since with the exception
of the simple example I'm showing here that is already available directly in boost-histogram, you will probably be
composing your own functions out of more than one builtin operation, you generally will not need the lambda here.

Manual compilation
^^^^^^^^^^^^^^^^^^

You can use strings to look up functions in the shared library:

.. code-block:: python

   def lookup(name):
       mylib = ctypes.CDLL("mylib.so")
       function = getattr(mylib, name)
       return ctypes.cast(function, ftype)

   bh.axis.Regular(10, 1, 4,
                   transform=bh.axis.transform.Function("my_log", "my_exp",
                                                         convert=lookup))
