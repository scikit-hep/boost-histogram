# What's new in boost-histogram

## Version 0.11
### Version 0.11.0

A release focused on preparing for the upcoming Hist 2.0 release.

#### User changes

* Arbitrary items can be set on an axis or histogram. [#450][], [#456][]
* Subclasses can customize the conversion procedure. [#456][]

#### Bug fixes

* Fixed reading pickles from boost-histogram 0.6-0.8 [#445][]
* Minor correctness fix [#446][]
* Accidental install of typing on Python 3.5+ fixed
* Scalar ND fill fixed [#453][]

#### Developer changes
* Updated to Boost 1.74 [#442][]
* CMake installs version.py now too [#449][]
* Updated setuptools infrastructure no longer requires NumPy [#451][]
* Some basic clang-tidy checks are now being run [#455][]


[#442]: https://github.com/scikit-hep/boost-histogram/pull/442
[#445]: https://github.com/scikit-hep/boost-histogram/pull/445
[#446]: https://github.com/scikit-hep/boost-histogram/pull/446
[#449]: https://github.com/scikit-hep/boost-histogram/pull/449
[#450]: https://github.com/scikit-hep/boost-histogram/pull/450
[#451]: https://github.com/scikit-hep/boost-histogram/pull/451
[#453]: https://github.com/scikit-hep/boost-histogram/pull/453
[#455]: https://github.com/scikit-hep/boost-histogram/pull/455
[#456]: https://github.com/scikit-hep/boost-histogram/pull/456


## Version 0.10

### Version 0.10.2

Quick fix for extra print statement in fill.

#### Bug fixes

* Fixed debugging print statement in fill. [#438][]

#### Developer changes

* Added CI/pre-commit check for print statements [#438][]
* Formatting CMakeLists now too [#439][]

[#438]: https://github.com/scikit-hep/boost-histogram/pull/438
[#439]: https://github.com/scikit-hep/boost-histogram/pull/439

### Version 0.10.1

Several fixes were made, mostly related to Weight storage histograms from Uproot 4.

#### Bug fixes

* Reduction on `h.axes.widths` supported again [#428][]
* WeightedSumView supports standard array operations [#432][]
* Operations shallow copy (non-copyable metadata supported) [#433][]
* Pandas Series as samples/weights supported [#434][]
* Support NumPy scalars in operations [#436][]

[#428]: https://github.com/scikit-hep/boost-histogram/pull/428
[#432]: https://github.com/scikit-hep/boost-histogram/pull/432
[#433]: https://github.com/scikit-hep/boost-histogram/pull/433
[#434]: https://github.com/scikit-hep/boost-histogram/pull/434
[#436]: https://github.com/scikit-hep/boost-histogram/pull/436

### Version 0.10.0

This version was released during PyHEP 2020. Several improvements were made to
usability when plotting and indexing.

#### User changes

* AxesTuple array now support operations via ArrayTuple [#414][]
* Support `sum` and `bh.rebin` without slice [#424][]
* Nicer error messages in some cases [#415][]
* Made a few properties hidden for accumulators that were not public [#418][]
* Boolean now supports reduction, faster compile [#422][]
* AxesTuple now available publicly for subprojects [#419][]

#### Bug fixes

* Histograms support operations with arrays, no longer take the first element only [#417][]

[#414]: https://github.com/scikit-hep/boost-histogram/pull/414
[#414]: https://github.com/scikit-hep/boost-histogram/pull/414
[#415]: https://github.com/scikit-hep/boost-histogram/pull/415
[#417]: https://github.com/scikit-hep/boost-histogram/pull/417
[#418]: https://github.com/scikit-hep/boost-histogram/pull/418
[#419]: https://github.com/scikit-hep/boost-histogram/pull/419
[#422]: https://github.com/scikit-hep/boost-histogram/pull/422
[#424]: https://github.com/scikit-hep/boost-histogram/pull/424

## Version 0.9
### Version 0.9.0

This version was released just before PyHEP 2020. Several important fixes were made,
along with a few new features to better support downstream projects.

#### User changes

* `metadata` supported and propagated on Histograms (slots added) [#403][]
* Added `dd=True` option in `to_numpy` [#406][]
* Deprecated `cpp` module removed [#402][]

#### Developer changes

* Subclasses can override axes generation [#401][]
* `[dev]` extra now installs `pytest` [#401][]

#### Bug fixes

* Fix `numpy.histogramdd` return structure [#406][]
* Travis deploy multi-arch fixes [#399][]
* Selecting on a bool axes supports 2D+ histograms [#398][]
* Warnings fixed on NumPy 1.19+ [#404][]

[#398]: https://github.com/scikit-hep/boost-histogram/pull/398
[#399]: https://github.com/scikit-hep/boost-histogram/pull/399
[#401]: https://github.com/scikit-hep/boost-histogram/pull/401
[#402]: https://github.com/scikit-hep/boost-histogram/pull/402
[#403]: https://github.com/scikit-hep/boost-histogram/pull/403
[#406]: https://github.com/scikit-hep/boost-histogram/pull/406

## Version 0.8
### Version 0.8.0

This version was released just before SciPy 2020 and Boost 1.74. Highlights
include better accumulator views, simpler summing, better NumPy and Pandas
compatibility, and sums on growing axes. Lots of backend work,
including a new wheel building system, internal changes and better reliance
on Boost.Histogram's C++ tools for actions like cropping.

#### User changes

* Weighted histogram cells can now be assigned directly from iterables [#375][]
* Weighted views can be summed and added [#368][]
* Sum is now identical to the built-in sum function [#365][]
* Adding growing axis is better supported [#358][]
* Slicing an AxesTuple now keeps the type [#384][]
* `ndim` replaces `rank` for NumPy compatibility [#385][]
* Any array-like supported in fill [#391][], any iterable can be used for Categories [#392][]
* Added Boolean axes, from Boost.Histogram 1.74 [#390][]
* Division between histograms is supported [#393][]
* More deprecated functionality removed

#### Bug fixes

* Support older versions of [CloudPickle][] (issue also fixed upstream) [#343][]
* Drop extra printout [#338][]
* Throw an error instead of returning an incorrect result in more places [#386][]

#### Developer changes

* Update Boost to 1.73 [#359][], pybind11 to 2.5.0 [#351][], Boost.Histogram to pre-1.74 [#388][]
* Cropping no longer uses workaround [#373][]
* Many more checks added to [`pre-commit`][] [#366][]
* Deprecating `cpp` interface [#391][]
* Wheelbuilding migrated to [`cibuildwheel`][] and GHA [#361][]

[CloudPickle]: https://github.com/cloudpipe/cloudpickle
[`cibuildwheel`]: https://cibuildwheel.readthedocs.io/en/stable
[`pre-commit`]: https://pre-commit.com

[#338]: https://github.com/scikit-hep/boost-histogram/pull/338
[#340]: https://github.com/scikit-hep/boost-histogram/pull/340
[#343]: https://github.com/scikit-hep/boost-histogram/pull/343
[#351]: https://github.com/scikit-hep/boost-histogram/pull/351
[#358]: https://github.com/scikit-hep/boost-histogram/pull/358
[#359]: https://github.com/scikit-hep/boost-histogram/pull/359
[#361]: https://github.com/scikit-hep/boost-histogram/pull/361
[#365]: https://github.com/scikit-hep/boost-histogram/pull/365
[#366]: https://github.com/scikit-hep/boost-histogram/pull/366
[#368]: https://github.com/scikit-hep/boost-histogram/pull/368
[#373]: https://github.com/scikit-hep/boost-histogram/pull/373
[#375]: https://github.com/scikit-hep/boost-histogram/pull/375
[#384]: https://github.com/scikit-hep/boost-histogram/pull/384
[#385]: https://github.com/scikit-hep/boost-histogram/pull/385
[#386]: https://github.com/scikit-hep/boost-histogram/pull/386
[#388]: https://github.com/scikit-hep/boost-histogram/pull/388
[#390]: https://github.com/scikit-hep/boost-histogram/pull/390
[#391]: https://github.com/scikit-hep/boost-histogram/pull/391
[#392]: https://github.com/scikit-hep/boost-histogram/pull/392
[#393]: https://github.com/scikit-hep/boost-histogram/pull/393


## Version 0.7
### Version 0.7.0

This version removes deprecated functionality, and has several backend
improvements. The most noticeable user-facing change is the multithreaded fill
feature, which can enable significant speedups when you have a dataset that is
much larger than the number of bins in your histogram and have free cores to
use. Several small bugs have been fixed.

#### User changes

* Added `threads=` keyword to `.fill` and NumPy functions; 0 for automatic, default is 1 [#325][]
* `.metadata` is now settable directly from the AxesTuple [#303][]
* Deprecated items from 0.5.x now dropped [#301][]
* `cpp` mode updates and fixes [#317][]

#### Bug fixes

* Dict indexing is now identical to positional indexing, fixes "picking" axes in dict [#320][]
* Passing `samples=None` is now always allowed in `.fill` [#325][]

#### Developer changes

* Build system update, higher requirements for developers (only) [#314][]
    * Version is now obtained from `setuptools_scm`, no longer stored in repo
* Removed `futures` requirement for Python 2 tests
* Updated Boost.Histogram, cleaner code with fewer workarounds

[#301]: https://github.com/scikit-hep/boost-histogram/pull/301
[#303]: https://github.com/scikit-hep/boost-histogram/pull/303
[#314]: https://github.com/scikit-hep/boost-histogram/pull/314
[#317]: https://github.com/scikit-hep/boost-histogram/pull/317
[#320]: https://github.com/scikit-hep/boost-histogram/pull/320
[#325]: https://github.com/scikit-hep/boost-histogram/pull/325


## Version 0.6

### Version 0.6.2

Common analysis tasks are now better supported. Much more complete
documentation. Now using development branch of Boost.Histogram again.

#### Bug fixes

* Fix sum over category axes in indexing [#298][]
* Allow single category item selection [#298][]
* Allow slicing on axes without flow bins [#288][], [#300][]
* Sum repr no longer throws error [#293][]

#### Developer changes

* Now using scikit-hep/azure-wheel-helpers via subtree [#292][]

[#288]: https://github.com/scikit-hep/boost-histogram/pull/288
[#292]: https://github.com/scikit-hep/boost-histogram/pull/292
[#293]: https://github.com/scikit-hep/boost-histogram/pull/293
[#298]: https://github.com/scikit-hep/boost-histogram/pull/298
[#300]: https://github.com/scikit-hep/boost-histogram/pull/300

### Version 0.6.1

Examples and notebooks are now up to date with the current state of the
library. Using Boost 1.72 release.

#### User changes

* Slices and single values can be mixed in indexing [#279][]
* UHI locators supported on axes [#280][]

#### Bug fixes

* Properties on accumulator views now resolve correctly [#273][]
* Division of a histogram by a number is supported again [#278][]
* Setting a histogram with length one slice fixed [#279][]
* Numpy functions now work with Numpy ints in `bins=` [#282][]
* In-place addition avoids a copy [#284][]

[#273]: https://github.com/scikit-hep/boost-histogram/pull/273
[#278]: https://github.com/scikit-hep/boost-histogram/pull/278
[#279]: https://github.com/scikit-hep/boost-histogram/pull/279
[#280]: https://github.com/scikit-hep/boost-histogram/pull/280
[#282]: https://github.com/scikit-hep/boost-histogram/pull/282
[#284]: https://github.com/scikit-hep/boost-histogram/pull/284



### Version 0.6.0


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

#### Bug fixes

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


## Version 0.5

### Version 0.5.2

#### User changes:

* `bh.loc` supports an offset [#164][]
* Nicer reprs in several places [#167][]
* Deprecate `.at` and `.axis` [#170][]

#### Bug fixes:

* Use relative paths in setup.py to avoid resolving WSL paths on Windows [#162][], [#163][]
* Better pybind11 support for Python 3.8 [#168][]

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


### Version 0.5.0

First beta release and beginning of the changelog.

#### Known issues:

* Unlimited storage does not support pickling or classic multiprocessing
* Some non-simple storages do not support some forms of access, like `.view`
* Indexing and the array versions (such as centers) are incomplete and subject to change
* The numpy module is provisional and subject to change
* Docstrings and signatures will improve in later versions (especially on Python 3)
