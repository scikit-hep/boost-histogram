"""
This test module does simple smoke checks to make sure boost_histogram compiled
correctly. Run with `python -m boost_histogram.test`.
"""

from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main("boost_histogram.test", warnings="error")
