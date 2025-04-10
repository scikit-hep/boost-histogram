# To regenerate API docs:
# nox -s build_api_docs

from __future__ import annotations

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import contextlib
import importlib.metadata
import shutil
import sys
from pathlib import Path

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config
# -- Path setup --------------------------------------------------------------


DIR = Path(__file__).parent
BASEDIR = DIR.parent

sys.path.append(str(BASEDIR / "src"))

# -- Project information -----------------------------------------------------

project = "boost_histogram"
copyright = "2020, Henry Schreiner, Hans Dembinski"
author = "Henry Schreiner, Hans Dembinski"

with contextlib.suppress(ModuleNotFoundError):
    version = importlib.metadata.version("boost_histogram")
    # passed if no version (latest/git hash)


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
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

# Intersphinx setup
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

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
    outer_nb = BASEDIR / "notebooks"
    inner_nb = DIR / "notebooks"
    notebooks = outer_nb.glob("*.ipynb")
    for notebook in notebooks:
        shutil.copy(notebook, inner_nb / notebook.name)

    outer_cont = BASEDIR / ".github"
    inner_cont = DIR
    contributing = "CONTRIBUTING.md"
    shutil.copy(outer_cont / contributing, inner_cont / "contributing.md")


def clean_up(app, exception):
    inner_nb = DIR / "notebooks"
    for notebook in inner_nb.glob("*.ipynb"):
        notebook.unlink()

    inner_cont = DIR
    (inner_cont / "contributing.md").unlink()


def setup(app):
    # Copy the notebooks in
    app.connect("builder-inited", prepare)

    # Clean up the generated notebooks
    app.connect("build-finished", clean_up)
