#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pickle
from pathlib import Path

import typer

import boost_histogram as bh


def make_pickle(
    output: Path = typer.Argument(..., exists=False), *, protocol: int = 2  # noqa: B008
):
    """
    Make a pickle file with the current boost-histogram for use in tests.
    """

    VER = tuple(map(int, bh.__version__.split(".")))

    h1 = bh.Histogram(bh.axis.Regular(31, -15, 15))
    h2 = bh.Histogram(
        bh.axis.Integer(0, 5, metadata={"hello": "world"}), storage=bh.storage.Weight()
    )

    if VER >= (0, 9, 0):
        h2.metadata = "foo"

    h3 = bh.Histogram(
        bh.axis.StrCategory([], growth=True), bh.axis.Variable([0, 1, 2, 3])
    )

    h1.fill([-5, 1, 1, 2])
    h2.fill([1, 2, 2, 3, 3, 3])
    h3.fill(["one", "two", "two"], [3, 2, 1])

    with output.open("wb") as f:
        pickle.dump(
            {"h1": h1, "h2": h2, "h3": h3, "version": bh.__version__}, f, protocol
        )


if __name__ == "__main__":
    typer.run(make_pickle)
