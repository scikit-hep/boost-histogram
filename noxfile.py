from __future__ import annotations

import argparse
import shutil

import nox

ALL_PYTHONS = ["3.7", "3.8", "3.9", "3.10", "3.11"]

nox.options.sessions = ["lint", "tests"]


@nox.session(python=ALL_PYTHONS)
def tests(session: nox.Session) -> None:
    """
    Run the unit and regular tests.
    """

    shutil.rmtree("build", ignore_errors=True)
    session.install(".[test]")
    session.run("pytest", *session.posargs)


@nox.session
def hist(session: nox.Session) -> None:
    """
    Run Hist's test suite
    """
    shutil.rmtree("build", ignore_errors=True)
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
    Build the docs. Pass "--serve" to serve. Pass "-b linkcheck" to check links.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="Serve after building")
    parser.add_argument(
        "-b", dest="builder", default="html", help="Build target (default: html)"
    )
    args, posargs = parser.parse_known_args(session.posargs)

    if args.builder != "html" and args.serve:
        session.error("Must not specify non-HTML builder with --serve")

    extra_installs = ["sphinx-autobuild"] if args.serve else []

    session.chdir("docs")
    session.install("-r", "requirements.txt", *extra_installs)

    if args.builder == "linkcheck":
        session.run(
            "sphinx-build", "-b", "linkcheck", ".", "_build/linkcheck", *posargs
        )
        return

    shared_args = (
        "-n",  # nitpicky mode
        "-T",  # full tracebacks
        f"-b={args.builder}",
        ".",
        f"_build/{args.builder}",
        *posargs,
    )

    if args.serve:
        session.run("sphinx-autobuild", *shared_args)
    else:
        session.run("sphinx-build", "--keep-going", *shared_args)


@nox.session
def build_api_docs(session: nox.Session) -> None:
    """
    Build (regenerate) API docs.
    """

    session.install(".", "-r", "docs/requirements.txt")
    session.chdir("docs")
    session.run(
        "sphinx-apidoc",
        "-o",
        "api/",
        "--no-toc",
        "--template",
        "template/",
        "--force",
        "--module-first",
        "../src/boost_histogram",
    )


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

    session.install("pylint==2.17.*")
    session.install(".")
    session.run("pylint", "src", *session.posargs)


@nox.session
def make_pickle(session: nox.Session) -> None:
    """
    Make a pickle file for this version
    """
    session.install(".[dev]")
    session.run("python", "tests/pickles/make_pickle.py", *session.posargs)
