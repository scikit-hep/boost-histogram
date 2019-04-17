#!/bin/bash
set -e -x

export NPY_NUM_BUILD_JOBS=4

# Collect the pythons
pys=(/opt/python/*/bin)

# Filter out Python 3.4
pys=(${pys[@]//*34*/})

# Compile wheels
for PYBIN in "${pys[@]}"; do
    "${PYBIN}/pip" install -r /io/dev-requirements.txt
    "${PYBIN}/pip" wheel /io/ -w wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/boost_histogram-*.whl; do
    auditwheel repair --plat $PLAT "$whl" -w /io/wheelhouse/
done

# Install packages and test
for PYBIN in "${pys[@]}"; do
    "${PYBIN}/pip" install boost_histogram --no-index -f /io/wheelhouse
    (cd /io/tests && "${PYBIN}/python" -m pytest)
done
