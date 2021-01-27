# -*- coding: utf-8 -*-
# To regenerate API docs:
# sphinx-apidoc -o api ../src/boost_histogram -M -f -t template/

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
import shutil
import sys

# Docs require Python 3.6+ to generate
from pathlib import Path

DIR = Path(__file__).parent.resolve()
BASEDIR = DIR.parent

sys.path.append(str(BASEDIR / "src"))

# -- Project information -----------------------------------------------------

project = "boost_histogram"
copyright = "2020, Henry Schreiner, Hans Dembinski"
author = "Henry Schreiner, Hans Dembinski"

# It is better to use pkg_resources, but we can't build on RtD
from pkg_resources import DistributionNotFound, get_distribution

try:
    version = get_distribution("boost_histogram").version
except DistributionNotFound:
    pass  # No version (latest/git hash)


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "recommonmark",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "nbsphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    "api/boost_histogram.*.rst",
]

# Read the Docs needs this explicitly listed.
master_doc = "index"

# -- Options for Notebook input ----------------------------------------------

html_logo = "_images/BoostHistogramPythonLogo.png"
html_title = "boost-histogram docs"

nbsphinx_execute = "never"  # Can change to auto

highlight_language = "python3"

nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'png2x'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"

# Config for the Sphinx book

html_baseurl = "https://boost-histogram.readthedocs.io/en/latest/"


html_theme_options = {
    "home_page_in_toc": True,
    "repository_url": "https://github.com/scikit-hep/boost-histogram",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []  # _static

# Simpler docs (no build required)

autodoc_mock_imports = ["boost_histogram._core"]


def prepare(app):
    outer = BASEDIR / "notebooks"
    inner = DIR / "notebooks"
    notebooks = outer.glob("*.ipynb")

    for notebook in notebooks:
        shutil.copy(notebook, inner / notebook.name)


def clean_up(app, exception):
    inner = DIR / "notebooks"
    for notebook in inner.glob("*.ipynb"):
        notebook.unlink()


def setup(app):

    # Copy the notebooks in
    app.connect("builder-inited", prepare)

    # Clean up the generated notebooks
    app.connect("build-finished", clean_up)
