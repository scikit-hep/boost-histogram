# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

from ..accumulators import (
    Sum as sum,
    Mean as mean,
    WeightedSum as weighted_sum,
    WeightedMean as weighted_mean,
)

del absolute_import, division, print_function

__all__ = ("sum", "mean", "weighted_sum", "weighted_mean")


# These will have the original module locations and original names.
