# /// script
# dependencies = ["nox>=2025.2.9"]
# ///

from __future__ import annotations

import argparse

import nox

nox.needs_version = ">=2025.2.9"
nox.options.default_venv_backend = "uv|virtualenv"


@nox.session
def tests(session: nox.Session) -> None:
    """
    Run the unit and regular tests.
    """
    opts = (
        ["--reinstall-package=boost-histogram"] if session.venv_backend == "uv" else []
    )
    args = session.posargs or ["-n", "auto"]
    pyproject = nox.project.load_toml("pyproject.toml")
    session.install(*nox.project.dependency_groups(pyproject, "test"))
    session.install("-v", ".", *opts, silent=False)
    session.run("pytest", *args)


@nox.session(default=False)
def hist(session: nox.Session) -> None:
    """
    Run Hist's test suite
    """
    args = session.posargs or ["-n", "auto"]
    pyproject = nox.project.load_toml("pyproject.toml")
    session.install(".")
    tmpdir = session.create_tmp()
    session.chdir(tmpdir)
    session.run("git", "clone", "https://github.com/scikit-hep/hist", external=True)
    session.chdir("hist")
    session.install(".", *nox.project.dependency_groups(pyproject, "test", "plot"))
    session.run("pip", "list")
    session.run("pytest", *args)


@nox.session(reuse_venv=True, default=False)
def docs(session: nox.Session) -> None:
    """
    Build the docs. Pass --non-interactive to avoid serving. Pass "-b linkcheck" to check links.
    """
    pyproject = nox.project.load_toml("pyproject.toml")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", dest="builder", default="html", help="Build target (default: html)"
    )
    args, posargs = parser.parse_known_args(session.posargs)
    serve = args.builder == "html" and session.interactive

    extra_installs = ["sphinx-autobuild"] if serve else []
    session.install(*nox.project.dependency_groups(pyproject, "docs"), *extra_installs)

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


@nox.session(default=False)
def build_api_docs(session: nox.Session) -> None:
    """
    Build (regenerate) API docs.
    """
    pyproject = nox.project.load_toml("pyproject.toml")

    session.install(*nox.project.dependency_groups(pyproject, "docs"))
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

    session.install("pylint==3.3.*")
    session.install("-e.")
    session.run("pylint", "boost_histogram", *session.posargs)


@nox.session(default=False)
def make_pickle(session: nox.Session) -> None:
    """
    Make a pickle file for this version
    """
    pyproject = nox.project.load_toml("pyproject.toml")
    session.install(".", *nox.project.dependency_groups(pyproject, "dev"))
    session.run("python", "tests/pickles/make_pickle.py", *session.posargs)


if __name__ == "__main__":
    nox.main()
