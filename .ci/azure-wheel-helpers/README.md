## Azure Wheel Helpers

This repository holds a collection of wheel helpers designed by the
[Scikit-HEP][] project to build Python Wheels on [Azure DevOps][]. This is
designed for packages that require building; if you have a pure-Python project,
producing a universal wheel is trivial without this helper collection. This
collection assumes some standard paths and procedures, though *some* of them
can be customized.

Azure provides manual pipeline triggering and release pipelines, making it
slighly better suited for this than GitHub Actions, though otherwise they are
*very* similar.

### Supported platforms and caveats

TLDR: Python 2.7, 3.6, 3.7, and 3.8  on all platforms, along with 3.5 on Linux.

| System | Arch | Python versions |
|---------|-----|------------------|
| SDist (all) | all |  any (non-binary distribution) |
| ManyLinux1 | 64 & 32-bit | 2.7, 3.5, 3.6, 3.7, 3.8 |
| ManyLinux2010 | 64-bit | 2.7, 3.5, 3.6, 3.7, 3.8 |
| macOS 10.9+ | 64-bit | 2.7, 3.6, 3.7, 3.8 |
| Windows | 64 & 32-bit | 2.7, 3.6, 3.7, 3.8 |

* Linux: Python 3.4 is not supported because Numpy does not support it either.
* manylinux1: Optional support for GCC 9.1 using docker image; should work but
  can't be called directly other compiled extensions unless they do the same
  thing (think that's the main caveat). Supporting 32 bits because it's there
  for Numpy and PPA for now.
* manylinux2010: Requires pip 10+ and a version of Linux newer than 2010. This
  is very new technology. 64-bit only. Eventually this will become the
  preferred (and then only) way to produce Linux wheels. Optional modern GCC
  image available.
* MacOS: Uses the dedicated 64 bit 10.9+ Python.org builds. We are not
  supporting 3.5 because those no longer provide binaries (could use 32+64 fat
  10.6+ but really force to 10.9+, but will not be added unless there is a need
  for it).
* Windows: PyBind11 requires compilation with a newer copy of Visual Studio
  than Python 2.7's Visual Studio 2008; you need to have the [Visual Studio
  2015 distributable][msvc2015] installed (the dll is included in 2017 and
  2019, as well).

[msvc2017]: https://www.microsoft.com/en-us/download/details.aspx?id=48145

### Usage

> Azure does not recognize git submodules during the configure phase. Therefore, we are using git subtree instead.

This repository should reside in `/.ci` in your project. To add it:

```bash
git subtree add --prefix .ci/azure-wheel-helpers git@github.com:scikit-hep/azure-wheel-helpers.git master --squash
```

You should make a copy of the template pipeline and make local edits:

```bash
cp .ci/azure-wheel-helpers/azure-pipeline-build.yml .ci/azure-pipeline-build.yml
```

Make sure you enable this path in Azure as the pipeline. See [the post here][iscinumpy/wheels] for more details.

You must set the variables at the top of this file, and remove any configurations (like Windows) that you do not support:

```yaml
variables:
  package_name: my_package    # This is the output name, - is replaced by _
  many_linux_base: "quay.io/pypa/manylinux1_" # Could also be "skhep/manylinuxgcc-"
  dev_requirements_file: .ci/azure-wheel-helpers/empty-requirements.txt
  test_requirements_file: .ci/azure-wheel-helpers/empty-requirements.txt
  MACOSX_DEPLOYMENT_TARGET: 10.9
```

You can adjust the rest of the template as needed. If you need a non-standard
procedure, you can change the target of the `template` inputs to a local file.
You must have a `test_requirments` file, as the manylinux wheel install test
does not pull requirements when testing, and at least pytest is required.


#### Updates

To update, run:

```bash
git subtree pull --prefix .ci/azure-wheel-helpers git@github.com:scikit-hep/azure-wheel-helpers.git master --squash
```

If you make changes inside the folder and want to contribute back, run:

```bash
git subtree push --prefix=.ci/azure-wheel-helpers git@github.com:scikit-hep/azure-wheel-helpers.git my_fixup_branch
```

As always, you can make a remote to shorten these commands.


### Common needs

#### Using numpy with Cython

If you build with Cython, you will need to require an older version of Numpy.
Either place this in your  `dev_requirements_file` (classic builds) or your
`pyproject.toml` (PEP 517 builds):

```
numpy==1.11.3; python_version<="3.5"
numpy==1.12.1; python_version=="3.6"
numpy==1.14.5; python_version=="3.7"
numpy==1.17.3; python_version>="3.8"
```

(Note: most of Scikit-HEP officially requires 1.13.3+, so you can simplify this
with a single `<='3.6'`)

#### Using PEP 517 builds

For PEP 517 builds, you need to have a pyproject.toml file. Then, for PIP > 10,
the build happens in a custom environment that has *only* the packages you
request. It replaces the deprecated and mostly non-functional `setup_requires`
in setup.py, and even lets you select a build system other than setuptools. If
you just use it as a replacement for `setup_requires`, you can still support
pip < 10; users will just have to manually install the requirements (usually
Numpy) beforehand. Here's an example of a Cython PEP 517 build:

```toml
[build-system]
requires = [
    "setuptools>=18.0",
    "wheel",
    "Cython>=0.29.13",
    "numpy==1.13.3; python_version<='3.6'",
    "numpy==1.14.5; python_version=='3.7'",
    "numpy==1.17.3; python_version>='3.8'",
]
```

Now, in `setup.py`, just `import numpy` and use it, no need to check to see if
it there, etc.

#### Using Numpy parallel compile

If you have numpy available, you can add parallel compiles trivially:

```python
# Use -j N or set the environment variable NPY_NUM_BUILD_JOBS
from numpy.distutils.ccompiler import CCompiler_compile
import distutils.ccompiler
distutils.ccompiler.CCompiler.compile = CCompiler_compile
```

#### Using Cython + Setuptools

Since `setuptools>=18.0`, you can now pass `.pyx` files directly as sources to
`Extension`, and they get Cythonized for you! You just need Cython installed. 

### License

Copyright (c) 2019, Henry Schreiner.

Distributed under the 3-clause BSD license, see accompanying file LICENSE
or <https://github.com/scikit-hep/azure-wheel-helpers> for details.


[Scikit-HEP]:   http://scikit-hep.org
[Azure DevOps]: https://dev.azure.com
[iscinumpy/wheels]: https://iscinumpy.gitlab.io/post/azure-devops-python-wheels/
[msvc2017]: https://www.microsoft.com/en-us/download/details.aspx?id=48145

