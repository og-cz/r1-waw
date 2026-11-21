# Copyright (c) 2026 OGCZ. All rights reserved. See LICENSE.
"""Pairwise dissimilarity representation: 17 values between two feature vectors (e.g. HOG)."""
from __future__ import annotations

import itertools

import numpy as np

EPS = 1e-9
NAMES = ["L1", "L2", "Linf", "cosine", "correlation", "canberra", "braycurtis", "absdiff_std", "absdiff_median",
         "absdiff_p90", "d_mean", "d_std", "d_max", "d_energy", "chi2", "hist_inter", "d_sum"]


def pair_descriptor(a, b) -> np.ndarray:
    """Symmetric 17-value dissimilarity of two feature vectors."""
    a = np.asarray(a, float); b = np.asarray(b, float); d = np.abs(a - b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    ac, bc = a - a.mean(), b - b.mean()
    return np.array([
        d.mean(), np.sqrt((d ** 2).sum()), d.max(),
        1 - float(a @ b) / (na * nb + EPS),
        1 - float(ac @ bc) / (np.linalg.norm(ac) * np.linalg.norm(bc) + EPS),
        float(np.sum(d / (np.abs(a) + np.abs(b) + EPS))),
        float(d.sum() / (np.abs(a + b).sum() + EPS)),
        d.std(), np.median(d), np.percentile(d, 90),
        abs(a.mean() - b.mean()), abs(a.std() - b.std()),
        abs(a.max() - b.max()), abs((a ** 2).sum() - (b ** 2).sum()),
        float(np.sum(d ** 2 / (np.abs(a) + np.abs(b) + EPS))),
        1 - float(np.minimum(np.abs(a), np.abs(b)).sum()) / (max(np.abs(a).sum(), np.abs(b).sum()) + EPS),
        abs(a.sum() - b.sum())], float)


def enrolment_pairs(F: np.ndarray, descriptor=pair_descriptor) -> np.ndarray:
    """All unordered pairs of the n reference vectors: (n(n-1)/2, 17). n = 12 gives 66 pairs."""
    return np.array([descriptor(F[i], F[j]) for i, j in itertools.combinations(range(len(F)), 2)])


def query_pairs(q: np.ndarray, F: np.ndarray, descriptor=pair_descriptor) -> np.ndarray:
    """The questioned vector against each reference: (n, 17)."""
    return np.array([descriptor(q, f) for f in F])
