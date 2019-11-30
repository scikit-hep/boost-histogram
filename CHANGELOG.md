## Version 0.6

This version fills out most of the remaining features missing from the 0.5.x
series.  You can now use all the storages without the original caveats; even
the accumulators can be accessed array-at-a-time without copy, pickled quickly,
and set array-at-a-time, as well.

The API has changed considerably, providing a more consistent experience in
Python. Most of the classic API still works in this release, but will issue a
warning and will be removed from the next release. Please use this release to
transition existing 0.5.x code to the new API.


#### User changes

* Histogram and Axis classes now follow PEP 8 naming scheme (`histogram`->`Histogram`, `regular`->`Regular`, `int`->`Int64` etc.) [#192][], [#255][]
* You can now view a histogram with accumulators, with property access such as `h.view().value` [#194][]
* Circular variable and integer axes added [#231][]
* Split Category into `StrCategory` and `IntCategory`, now allows empty categories when `growth=True` [#221][]
* `StrCategory` fills are safer and faster [#239][], [#244][]
* Added axes transforms [#192][]
* `Function(forward, inverse)` transform added, allowing ultra-fast C function pointer transforms [#231][]
* You can now set histogram contents directly [#250][]
* You can now sum over a range with endpoints [#185][]
* `h.axes` now has the functions from axis as well. [#183][]
* `bh.project` has become `bh.sum` [#185][]
* `.reduce(...)` and the reducers in `bh.algorithm` have been removed in favor of dictionary based UHI slicing [#259][]
* `bh.numpy` module interface updates, `histogram=bh.Histogram` replaces cryptic `bh=True`, and `density=True` is now supported in Numpy mode [#256][]
* Added `hist.copy()` [#218][] and `hist.shape` [#264][]
* Signatures are much nicer in Python 3 [#188][]
* Reprs are better, various properties like `__module__` are now set correctly [#200][]

#### Bug fixes:
* Unlimited and AtomicInt storages now allow single item access [#194][]
* `.view()` now no longer makes a copy [#194][]
* Fixes related to string category axis fills [#233][], [#230][]
* Axes are no longer copies, support setting metadata [#238][], [#246][]
* Pickling accumulator storages is now comparable in performance simple storages [#258][]

#### Developer changes

* The linux wheels are now 10-20x smaller [#229][]
* The hist/axis classes are now pure Python, with a C++ object inside [#183][]
* Most internal names changed, `core->_core`, etc. [#183][]
* The `uhi` module is now `tag`. [#183][]
* `boost_histogram.cpp as bh` provides C++ high-compatibility mode. [#183][]
* Indexing tags now use full UHI instead of workarounds [#185][]
* Removed log and sqrt special axes types[#231][]
* Family and registration added, new casting system [#200][]


[#183]: https://github.com/scikit-hep/boost-histogram/pull/183
[#185]: https://github.com/scikit-hep/boost-histogram/pull/185
[#188]: https://github.com/scikit-hep/boost-histogram/pull/188
[#192]: https://github.com/scikit-hep/boost-histogram/pull/192
[#194]: https://github.com/scikit-hep/boost-histogram/pull/194
[#200]: https://github.com/scikit-hep/boost-histogram/pull/200
[#218]: https://github.com/scikit-hep/boost-histogram/pull/218
[#221]: https://github.com/scikit-hep/boost-histogram/pull/221
[#229]: https://github.com/scikit-hep/boost-histogram/pull/229
[#230]: https://github.com/scikit-hep/boost-histogram/pull/230
[#231]: https://github.com/scikit-hep/boost-histogram/pull/231
[#233]: https://github.com/scikit-hep/boost-histogram/pull/233
[#238]: https://github.com/scikit-hep/boost-histogram/pull/238
[#239]: https://github.com/scikit-hep/boost-histogram/pull/239
[#244]: https://github.com/scikit-hep/boost-histogram/pull/244
[#246]: https://github.com/scikit-hep/boost-histogram/pull/246
[#250]: https://github.com/scikit-hep/boost-histogram/pull/250
[#255]: https://github.com/scikit-hep/boost-histogram/pull/255
[#256]: https://github.com/scikit-hep/boost-histogram/pull/256
[#258]: https://github.com/scikit-hep/boost-histogram/pull/258
[#259]: https://github.com/scikit-hep/boost-histogram/pull/259
[#264]: https://github.com/scikit-hep/boost-histogram/pull/264


### Version 0.5.2

#### User changes:

* `bh.loc` supports an offset [#164][]
* Nicer reprs in several places [#167][]
* Deprecate `.at` and `.axis` [#170][]

#### Bug fixes:

* Use relative paths in setup.py to avoid resolving WSL paths on Windows [#162][], [#163][]
* Better Pybind11 support for Python 3.8 [#168][]

#### Developer changes:

* Serialization code shared with Boost.Histogram [#166][]
* Avoid unused PEP 517 isolation for now [#171][] (may return with proper PEP 518 support eventually)


[#162]: https://github.com/scikit-hep/boost-histogram/pull/162
[#163]: https://github.com/scikit-hep/boost-histogram/pull/163
[#164]: https://github.com/scikit-hep/boost-histogram/pull/164
[#166]: https://github.com/scikit-hep/boost-histogram/pull/166
[#167]: https://github.com/scikit-hep/boost-histogram/pull/167
[#168]: https://github.com/scikit-hep/boost-histogram/pull/168
[#170]: https://github.com/scikit-hep/boost-histogram/pull/170
[#171]: https://github.com/scikit-hep/boost-histogram/pull/171


### Version 0.5.1

#### User changes:

* Removed the `bh.indexed`/`h.indexed` iterator [#150][]
* Added `.axes` AxisTuple, with direct access to properties [#150][]
* Cleaned up tab completion in IPython [#150][]

#### Bug fixes:

* Fixed a bug in the sdist missing Boost.Variant2 [#154][]
* Fixed filling on strided inputs [#158][]



[#150]: https://github.com/scikit-hep/boost-histogram/pull/150
[#154]: https://github.com/scikit-hep/boost-histogram/pull/154
[#158]: https://github.com/scikit-hep/boost-histogram/pull/158
[#159]: https://github.com/scikit-hep/boost-histogram/pull/159


## Version 0.5

First beta release and beginning of the changelog.

#### Known issues:

* Unlimited storage does not support pickling or classic multiprocessing
* Some non-simple storages do not support some forms of access, like `.view`
* Indexing and the array versions (such as centers) are incomplete and subject to change
* The numpy module is provisional and subject to change
* Docstrings and signatures will improve in later versions (especially on Python 3)

