# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import re

import sys

DIR = os.path.abspath(os.path.dirname(__file__))
BASEDIR = os.path.abspath(os.path.dirname(DIR))
sys.path.append(BASEDIR)

# -- Project information -----------------------------------------------------

project = "boost_histogram"
copyright = "2019, Henry Schreiner, Hans Dembinski"
author = "Henry Schreiner, Hans Dembinski"


# The full version, including alpha/beta/rc tags
def get_version(version_file):
    RE_VERSION = re.compile(r"""^__version__\s*=\s*['"]([^'"]*)['"]""", re.MULTILINE)
    with open(version_file) as f:
        contents = f.read()
    mo = RE_VERSION.search(contents)
    if not mo:
        raise RuntimeError("Unable to find version string in {}.".format(version_file))

    return mo.group(1)


release = get_version(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "boost_histogram", "version.py")
    )
)


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["recommonmark", "sphinx.ext.autodoc", "sphinx.ext.napoleon"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".env"]

# Read the Docs needs this explicitly listed.
master_doc = "index"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []  # _static

# Simpler docs (no build required)

autodoc_mock_imports = ["boost_histogram._core"]
