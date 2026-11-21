# Copyright (c) 2026 OGCZ. All rights reserved. See LICENSE.
"""R1-WAW: Rank-One Writer-Adaptive Whitening."""
from .core import BETA, R1WAW, PooledYardstick, inv_sqrt, r1waw_metric
from .descriptor import enrolment_pairs, pair_descriptor, query_pairs

__all__ = ["BETA", "PooledYardstick", "R1WAW", "r1waw_metric", "inv_sqrt", "pair_descriptor", "enrolment_pairs",
           "query_pairs"]
__version__ = "0.2.0"
