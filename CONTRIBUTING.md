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

While developers often work in CMake, the "correct" way to develop a python
package is in a virtual environment. This is how you would set one up with
Python 3:

```bash
python3 -m venv .env
source ./.env/bin/activate
pip install numpy ipykernel pytest-sugar numba matplotlib
python -m ipykernel install --user --name boost-hist
pip install -e .[test]
deactivate
```

Now, you can run run notebooks using your system jupyter lab, and it will list
the environment as available!


You can use setuptools (`setup.py`) or CMake 3.14+ to build the package. CMake
is preferable for most development, and setuptools is used for packaging on
PyPI. Make a build directory and run CMake. If you have a specific Python you
want to use, add `-DPYTHON_EXECUTABLE=$(which python)` or similar to the CMake
line. If you need help installing the latest CMake version, [visit this
page](https://cliutils.gitlab.io/modern-cmake/chapters/intro/installing.html);
one option is to use pip to install CMake.

```bash
cmake -S . -B build
cmake --build build -j4
```

## Testing

Run the unit tests (requires pytest and numpy). Use the `test` target from
anywhere, or use `ctest` from the build directory, like this:

```bash
ctest
# Directly running with Python pytest works too
python3 -m pytest
```

The tests require `numpy`, `pytest`, and `pytest-benchmark`. If you are using
Python 2, you will need `futures` as well.

To install using the pip method for development instead, run:

```bash
python3 -m venv .env
. .env/bin/activate
python -m pip install .[test]
```

You'll need to reinstall it if you want to rebuild.

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
pre-commit install # If you have Docker
pre-commit install --config=.pre-commit-nodocker.yaml # If you have clang-format 8
```

Now all changed files will be checked every time you git commit. You can check
it yourself (even without installing the hooks) using:

```bash
pre-commit run --all-files
```


## Common tasks


<details><summary>Updating dependencies (click to expand)</summary>

This will checkout new versions of the dependencies. Example given using the
fish shell.

```fish
for f in *
    cd $f
    git fetch
    git checkout boost-1.72.0 || echo "Not found"
    cd ..
end
```

</details>

<details><summary>Making a new release (click to expand)</summary>

- Finish merging open PRs that will go into VERSION
- Add most recent changes to the Changelog
    - Replace "in development" header with VERSION
- Bump version
    - Change in `boost_histogram/version.py`
    - Change banner in README to VERSION
- Sync master with develop through a PR
- Make sure the full wheel build runs on master without issues (will happen in
  previous step)
- Make the GitHub release in the GitHub UI. Copy the changelog entries and
  links for that version; this has to be done as part of the release and tag
  procedure for archival tools (Zenodo) to pick them up correctly. Titles
  should be roughly consistent. I like to give a little descriptive title after
  the version, though this was a massive release that touched almost every
  area.
    - Version tag should be `"v" + major + "." + minor + "." + patch`.
- This should trigger an Azure wheel build. Note the name of the build (should
  be the date plus a number)
- In the Azure web interface, go to release pipelines and click create release.
  Make sure the build it is pulling artifacts from matches the correct build
  (should always choose latest, which *should* be correct) See
  https://iscinumpy.gitlab.io/post/azure-devops-releases/ for details about
  Azure releases.
- Conda-forge will automatically make a PR to update a few hours later.


</details>
