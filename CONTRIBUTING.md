# Contributing to boost-histogram


## Building from source

This repository has dependencies in submodules. Check out the repository like this:

```bash
git clone --recursive https://github.com/scikit-hep/boost-histogram.git
cd boost-histogram
```


<details><summary>Faster version (click to expand)</summary>

```bash
git clone https://github.com/scikit-hep/boost-histogram.git
cd boost-histogram
git submodule update --init --depth 10
```

</details>

## Development environment

### Pip

While developers often work in CMake, the "correct" way to develop a python
package is in a virtual environment. This is how you would set one up with
Python 3:

```bash
python3 -m venv .env
source ./.env/bin/activate
pip install -ve .[all]
```

<details><summary>Optional: External Jupyter kernel (click to expand)</summary>

You can set up a kernel for external Jupyter then deactivate your environment:

```python
python -m ipykernel install --user --name boost-hist
deactivate
```

Now, you can run notebooks using your system JupyterLab, and it will list
the environment as available!
</details>

To rebuild, rerun `pip install -ve .` from the environment, if the commit has
changed, you will get a new build. Due to the `-e`, Python changes do not require
a rebuild.

### CMake

CMake is common for C++ development, and ties nicely to many C++ tools, like
IDEs. If you want to use it for building, you can. Make a build directory and
run CMake. If you have a specific Python you want to use, add
`-DPYTHON_EXECUTABLE=$(which python)` or similar to the CMake line. If you need
help installing the latest CMake version, [visit this
page](https://cliutils.gitlab.io/modern-cmake/chapters/intro/installing.html);
one option is to use pip to install CMake.


> Note: Since setuptools uses a subdirectory called `build`, it is *slightly*
> better to avoid making your CMake directory `build` as well. Also, you will
> often have multiple CMake directories (`build-release`, `build-debug`, etc.),
> so avoiding the descriptive name `build` is not a bad idea.

You have three options for running code in python:

1. Run from the build directory (only works with some commands, like `python -m
pytest`, and not others, like `pytest`
2. Add the build directory to your PYTHONPATH environment variable
3. Set `CMAKE_INSTALL_PREFIX` to your site-packages and install (recommended
for virtual environments).

Here is the recommendation for a CMake install:


```bash
python3 -m venv env_cmake
source ./env_cmake/bin/activate
pip install -r dev-requirements.txt
cmake -S . -B build-debug \
    -GNinja \
    -DCMAKE_INSTALL_PREFIX=$(python -c "import distutils.sysconfig; print(distutils.sysconfig.get_python_lib(plat_specific=False,standard_lib=False))")
cmake --build build-debug -j4
cmake --install build-debug # Option 3 only
```

Note that option 3 will require reinstalling if the python files change, while
options 1-2 will not if you have a recent version of CMake (symlinks are made).

This could be simplified if pybind11 supported the new CMake FindPython tools.

## Testing

Run the unit tests (requires pytest and NumPy).

```bash
python3 -m pytest
```


For CMake, you can also use the `test` target from anywhere, or use `python3 -m
pytest` or `ctest` from the build directory.

The build requires `setuptools_scm`. The tests require `numpy`, `pytest`, and
`pytest-benchmark`. `pytest-sugar` adds some nice formatting.

## Benchmarking

You can enable benchmarking with `--benchmark-enable` when running tests. You
can also run explicit performance tests with `scripts/performance_report.py`.

```bash
python3 -m pytest --benchmark-enable --benchmark-sort fullname
```

For example, if you want to benchmark before and after a change:

```bash
python3 -m pytest --benchmark-enable --benchmark-autosave
# Make change
python3 -m pytest --benchmark-enable --benchmark-autosave

pytest-benchmark compare 0001 0002 --sort fullname --histogram
```

Note, while the histogram option (`--histogram`) is nice, it does require
`pygal` and `pygaljs` to be installed. Feel free to leave it off if not needed.

</details>

## Formatting

Code should be well formatted; CI will check it and one of the authors can help
reformat your code. If you want to check it yourself, you should use
[`pre-commit`](https://pre-commit.com).

Just [install pre-commit](https://pre-commit.com/#install), probably using brew
on macOS or pip on other platforms, then run:

```bash
pre-commit install
```

Now all changed files will be checked every time you git commit. You can check
it yourself (even without installing the hooks) using:

```bash
pre-commit run --all-files
```

We do not check `check-manifest` every time locally, since it is slow. You can trigger
this manual check with:

```
pre-commit run --all-files --hook-stage manual check-manifest
```

Developers should update the pre-commit dependencies once in a while, you can
do this automatically with:

```bash
pre-commit autoupdate
```

> #### Note about skipping Docker
>
> Pre-commit uses docker to ensure a consistent run of clang-format. If you do
> not want to install/run Docker, you should use `SKIP=docker-clang-format`
> when running pre-commit, and instead run `clang-format -style=file -i
> <files>` yourself.

## Clang-Tidy

To run Clang tidy, the following recipe should work. Files will be modified in
place, so you can use git to monitor the changes.

```bash
docker run --rm -v $PWD:/pybind11 -it silkeh/clang:10
apt-get update && apt-get install python3-dev
cmake -S pybind11/ -B build -DCMAKE_CXX_CLANG_TIDY="$(which clang-tidy);-fix"
cmake --build build
```

Remember to build single-threaded if applying fixes!

## Include what you use

To run include what you use, install (`brew install include-what-you-use` on
macOS), then run:

```bash
cmake -S . -B build-iwyu -DCMAKE_CXX_INCLUDE_WHAT_YOU_USE=$(which include-what-you-use)
cmake --build build
```

## Common tasks


<details><summary>Updating dependencies (click to expand)</summary>

This will checkout new versions of the dependencies. Example given using the
fish shell.

```fish
for f in *
    cd $f
    git fetch
    git checkout boost-1.75.0 || echo "Not found"
    cd ..
end
```

</details>

<details><summary>Making a new release (click to expand)</summary>

- Finish merging open PRs that you want in the new version
- Add most recent changes to the `docs/CHANGELOG.md`
- Sync master with develop using `git checkout master; git merge develop --ff-only` and push
- Make sure the full wheel build runs on master without issues (will happen
  automatically on push to master)
- Make the GitHub release in the GitHub UI. Copy the changelog entries and
  links for that version; this has to be done as part of the release and tag
  procedure for archival tools (Zenodo) to pick them up correctly.
    - Title should be `"Version <version number>"`
    - Version tag should be `"v" + major + "." + minor + "." + patch`.
- GHA will build and send to PyPI for you when you release.
- Conda-forge will automatically make a PR to update within an hour or so, and
  it will merge automatically if it passes.


</details>

<details><summary>Making a compiler flamegraph (click to expand)</summary>

This requires LLVM 9+, and is based on [this post](https://aras-p.info/blog/2019/01/16/time-trace-timeline-flame-chart-profiler-for-Clang/).

```bash
brew install llvm         # macOS way to get clang-9
python3 -m venv .env_core # general environment (no install will be made)
. .env_core/bin/activate
pip install -r dev-requirements.txt
CXX="/usr/local/opt/llvm/bin/clang++" cmake -S . -B build-llvm \
    -DCMAKE_CXX_FLAGS="-ftime-trace" \
    -DPYTHON_EXECUTABLE=$(which python)
cmake --build build-llvm/
```

Now open a browser with [SpeedScope](https://www.speedscope.app), and load one of the files.

</details>

<details><summary>Adding a contributor (click to expand)</summary>

First, you need to install the [all contributor CLI](https://allcontributors.org/docs/en/cli/installation):

```bash
yarn add --dev all-contributors-cli
```

Then, you can add contributors:

```bash
yarn all-contributors add henryiii maintenance,code,doc
```

</details>
