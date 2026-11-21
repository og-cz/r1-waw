# Copyright (c) 2026 OGCZ. All rights reserved. See LICENSE.
"""Rank-One Writer-Adaptive Whitening (R1-WAW).

Phase A (once, development population):  PooledYardstick().fit(list of per-writer genuine pair matrices)
Phase B (per writer, genuine only):       R1WAW(yardstick).fit(that writer's genuine pairs)

beta is the adaptation strength (default 3); beta = 0 gives the shared whitening G.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-9        # added to the global scale of each dissimilarity value
EV_FLOOR = 1e-8   # eigenvalue floor in every inverse square root
BETA = 3.0        # adaptation strength


def inv_sqrt(S: np.ndarray, floor: float = EV_FLOOR) -> np.ndarray:
    """Symmetric inverse square root S^(-1/2), eigenvalues floored at `floor`."""
    w, V = np.linalg.eigh(S)
    return (V * np.maximum(w, floor) ** -0.5) @ V.T


class PooledYardstick:
    """Writer-independent constants (Phase A).

    Fitted on the genuine pairs of a development population: one (n_w, d) matrix per writer.
    scale_    (d,)   standard deviation of each dissimilarity value over all pairs (+ EPS)
    cov_      (d, d) pooled within-writer covariance of the standardised pairs
    eigvals_  (d,)   its eigenvalues, descending;  eigvals_[0] = lambda_1
    eigvecs_  (d, d) its eigenvectors as columns;  eigvecs_[:, 0] = u_1
    W_        (d, d) cov_^(-1/2), the shared whitening (the baseline "G")
    """

    def fit(self, writer_pairs: list[np.ndarray]) -> "PooledYardstick":
        self.scale_ = np.vstack(writer_pairs).std(0) + EPS
        self.cov_ = np.mean([self.writer_cov(P) for P in writer_pairs], axis=0)
        w, V = np.linalg.eigh(self.cov_)
        order = np.argsort(w)[::-1]
        self.eigvals_, self.eigvecs_ = w[order], V[:, order]
        self.W_ = inv_sqrt(self.cov_)
        return self

    def writer_cov(self, P: np.ndarray) -> np.ndarray:
        """Covariance of one writer's standardised, writer-centred pairs (biased, 1/n)."""
        Z = (P - P.mean(0)) / self.scale_
        return Z.T @ Z / len(Z)


def r1waw_metric(yardstick: PooledYardstick, sigma2: float, beta: float = BETA) -> np.ndarray:
    """W = Sigma_pool^(-1/2) + (s_eff^(-1/2) - lambda_1^(-1/2)) u_1 u_1^T,  s_eff = lambda_1 (sigma2 / lambda_1)^beta.

    u_1 is an eigenvector of Sigma_pool, so this is exactly inv_sqrt(Sigma_pool + (s_eff - lambda_1) u_1 u_1^T).
    """
    u1, lam1 = yardstick.eigvecs_[:, 0], yardstick.eigvals_[0]
    s_eff = lam1 * (max(sigma2, EV_FLOOR) / lam1) ** beta
    return yardstick.W_ + (s_eff ** -0.5 - lam1 ** -0.5) * np.outer(u1, u1)


class R1WAW:
    """Writer-adaptive metric (Phase B): one number per writer, estimated from genuine pairs only.

    sigma2_       the writer's variance along u_1:  u_1^T Sigma_w u_1
    consistency_  c_w = sqrt(sigma2_ / lambda_1)   (< 1: steadier than the average writer)
    W_            the writer's metric
    beta          adaptation strength (see module docstring)
    """

    def __init__(self, yardstick: PooledYardstick, beta: float = BETA):
        self.yardstick, self.beta = yardstick, float(beta)

    def fit(self, P: np.ndarray) -> "R1WAW":
        y = self.yardstick
        self.mean_ = P.mean(0)
        z = (P - self.mean_) / y.scale_
        self.sigma2_ = float(np.mean((z @ y.eigvecs_[:, 0]) ** 2))
        self.consistency_ = float(np.sqrt(self.sigma2_ / y.eigvals_[0]))
        self.W_ = r1waw_metric(y, self.sigma2_, self.beta)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Pairs X measured in the writer's space: ((X - mean) / scale) W."""
        return ((X - self.mean_) / self.yardstick.scale_) @ self.W_
