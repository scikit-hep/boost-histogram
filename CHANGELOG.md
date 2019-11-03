### IN DEVELOPMENT

#### User changes

* You can now sum over a range with endpoints [#185][]
* You can now access the functional regular axis directly, `regular_sqrt` becomes `regular.sqrt`, etc.. [#183][]
* `h.axis()` and `h.at()` are now completely removed unless you use the cpp version of histogram. [#183][]
* `h.axes` now has the functions from axis as well. [#183][]
* `bh.project` has become `bh.sum` [#185][]


#### Developer changes

* The hist/axis classes are now pure Python, with a C++ object inside [#183][]
* Most internal names changed, `core->_core`, etc. [#183][]
* The `uhi` module is now `tag`. [#183][]
* `boost_histogram.cpp as bh` provides C++ high-compatibility mode. [#183][]
* Indexing tags now use full UHI instead of workarounds [#185][]


[#183]: https://github.com/scikit-hep/boost-histogram/pull/183
[#185]: https://github.com/scikit-hep/boost-histogram/pull/185


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

