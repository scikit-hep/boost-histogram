"""
This test module does simple smoke checks to make sure boost_histogram compiled
correctly. Run with `python -m boost_histogram.test`.
"""

from __future__ import annotations

import pickle
import unittest

import boost_histogram as bh


class TestBoostHistogram(unittest.TestCase):
    def test_1d_histogram(self) -> None:
        h = bh.Histogram(bh.axis.Regular(10, 0, 1))
        h.fill([0.15, 0.25, 0.25, 0.35])
        self.assertEqual(h[0], 0)
        self.assertEqual(h[1], 1)
        self.assertEqual(h[2], 2)
        self.assertEqual(h[3], 1)
        self.assertEqual(h[4], 0)

    def test_2d_histogram(self) -> None:
        h = bh.Histogram(
            bh.axis.Integer(4, 10),
            bh.axis.StrCategory(["a", "b", "c"], overflow=False),
            storage=bh.storage.Weight(),
        )
        h.fill([5, 5, 5, 7], ["a", "b", "b", "c"])

        val = h[bh.loc(5), bh.loc("a")]
        assert isinstance(val, bh.accumulators.WeightedSum)
        self.assertEqual(val.value, 1)

        val = h[bh.loc(5), bh.loc("b")]
        assert isinstance(val, bh.accumulators.WeightedSum)
        self.assertEqual(val.value, 2)

        val = h[bh.loc(7), bh.loc("c")]
        assert isinstance(val, bh.accumulators.WeightedSum)
        self.assertEqual(val.value, 1)

        val = h[bh.loc(8), bh.loc("c")]
        assert isinstance(val, bh.accumulators.WeightedSum)
        self.assertEqual(val.value, 0)

    def test_pickle(self) -> None:
        h = bh.Histogram(
            bh.axis.Variable([0, 0.3, 0.4, 0.5, 1.0]), storage=bh.storage.Mean()
        )
        h.fill([0.15, 0.25, 0.25, 0.35], sample=[1, 2, 3, 4])
        pickled = pickle.dumps(h)
        unpickled = pickle.loads(pickled)
        self.assertEqual(h, unpickled)

    def test_sqrt_transform(self) -> None:
        a = bh.axis.Regular(10, 0, 10, transform=bh.axis.transform.sqrt)
        # Edges: 0. ,  0.1,  0.4,  0.9,  1.6,  2.5,  3.6,  4.9,  6.4,  8.1, 10.

        self.assertEqual(a.index(-100), 10)  # Always in overflow bin
        self.assertEqual(a.index(-1), 10)  # When transform is invalid
        self.assertEqual(a.index(0), 0)
        self.assertEqual(a.index(0.15), 1)
        self.assertEqual(a.index(0.5), 2)
        self.assertEqual(a.index(1), 3)
        self.assertEqual(a.index(1.7), 4)
        self.assertEqual(a.index(9), 9)
        self.assertEqual(a.index(11), 10)
        self.assertEqual(a.index(1000), 10)

        val = a.bin(0)
        assert isinstance(val, tuple)
        self.assertAlmostEqual(val[0], 0.0)

        val = a.bin(1)
        assert isinstance(val, tuple)
        self.assertAlmostEqual(val[0], 0.1)

        val = a.bin(2)
        assert isinstance(val, tuple)
        self.assertAlmostEqual(val[0], 0.4)


if __name__ == "__main__":
    unittest.main("boost_histogram.test", warnings="error")
