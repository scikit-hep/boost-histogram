from __future__ import annotations

import argparse
from pathlib import Path

import nox

nox.needs_version = ">=2024.3.2"
nox.options.default_venv_backend = "uv|virtualenv"
nox.options.sessions = ["lint", "tests"]


@nox.session
def tests(session: nox.Session) -> None:
    """
    Run the unit and regular tests.
    """

    session.install(".[test]")
    session.run("pytest", *session.posargs)


@nox.session
def hist(session: nox.Session) -> None:
    """
    Run Hist's test suite
    """
    session.install(".")
    tmpdir = session.create_tmp()
    session.chdir(tmpdir)
    session.run("git", "clone", "https://github.com/scikit-hep/hist", external=True)
    session.chdir("hist")
    session.install(".[test,plot]")
    session.run("pip", "list")
    session.run("pytest", *session.posargs)


@nox.session(reuse_venv=True)
def docs(session: nox.Session) -> None:
    """
    Build the docs. Pass --non-interactive to avoid serving. Pass "-b linkcheck" to check links.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", dest="builder", default="html", help="Build target (default: html)"
    )
    args, posargs = parser.parse_known_args(session.posargs)
    serve = args.builder == "html" and session.interactive

    extra_installs = ["sphinx-autobuild"] if serve else []
    session.install("-r", "docs/requirements.txt", *extra_installs)

    shared_args = (
        "-n",  # nitpicky mode
        "-T",  # full tracebacks
        f"-b={args.builder}",
        "docs",
        *(posargs or [f"docs/_build/{args.builder}"]),
    )

    if serve:
        session.run(
            "sphinx-autobuild", "--open-browser", "--ignore=docs/.build", *shared_args
        )
    else:
        session.run("sphinx-build", "--keep-going", *shared_args)


@nox.session
def build_api_docs(session: nox.Session) -> None:
    """
    Build (regenerate) API docs.
    """

    session.install(".", "-r", "docs/requirements.txt")
    session.run(
        "sphinx-apidoc",
        "-o",
        "docs/api/",
        "--no-toc",
        "--template",
        "docs/template/",
        "--force",
        "--module-first",
        "src/boost_histogram",
    )

    # add API docs of boost_histogram._internal.hist.Histogram after
    # the generation step
    with Path("docs/api/boost_histogram.rst").open("r+") as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i] == ".. automodule:: boost_histogram\n":
                lines[i] = ".. automodule:: boost_histogram._internal.hist\n"
                lines[i + 1] = "   :members: Histogram\n"
                break

        f.truncate(0)
        f.seek(0)
        f.writelines(lines)


@nox.session
def lint(session: nox.Session) -> None:
    """
    Run the linter.
    """
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files", *session.posargs)


@nox.session
def pylint(session: nox.Session) -> None:
    """
    Run pylint.
    """

    session.install("pylint==3.2.*")
    session.install("-e.")
    session.run("pylint", "boost_histogram", *session.posargs)


@nox.session
def make_pickle(session: nox.Session) -> None:
    """
    Make a pickle file for this version
    """
    session.install(".[dev]")
    session.run("python", "tests/pickles/make_pickle.py", *session.posargs)
