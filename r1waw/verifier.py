# Copyright (c) 2026 OGCZ. All rights reserved. See LICENSE.
"""Writer-dependent one-class verifier using R1-WAW (requires scikit-learn)."""
from __future__ import annotations

import numpy as np
from sklearn.svm import OneClassSVM

from .core import BETA, R1WAW, PooledYardstick
from .descriptor import enrolment_pairs, pair_descriptor, query_pairs

SHARPNESS = 3.0   # kernel sharpness; 1.0 equals scikit-learn's gamma="scale"


def kernel_gamma(X: np.ndarray, sharpness: float = SHARPNESS) -> float:
    """RBF width for the writer's transformed pairs: gamma = sharpness / (d * Var X)."""
    return sharpness / (X.shape[1] * X.var())


class Verifier:
    """Enrol a writer from genuine feature vectors only; score and decide questioned ones.

    score(q)  = mean OCSVM decision value of q's pairs with the references (higher = more genuine-like)
    decide(q) = verdict against a per-writer line from leave-one-out scores of the references:
                line = mean(loo) - alpha * sd(loo);  |score - line| <= zone * sd(loo)  ->  "uncertain"
    """

    def __init__(self, yardstick: PooledYardstick, beta: float = BETA, sharpness: float = SHARPNESS,
                 nu: float = 0.10, descriptor=pair_descriptor):
        self.yardstick, self.beta, self.sharpness, self.nu, self.descriptor = yardstick, beta, sharpness, nu, descriptor

    def _fit(self, F):
        P = enrolment_pairs(F, self.descriptor)
        metric = R1WAW(self.yardstick, self.beta).fit(P)
        X = metric.transform(P)
        svm = OneClassSVM(kernel="rbf", nu=self.nu, gamma=kernel_gamma(X, self.sharpness)).fit(X)
        return metric, svm

    def _score(self, metric, svm, q, F):
        return float(svm.decision_function(metric.transform(query_pairs(q, F, self.descriptor))).mean())

    def enrol(self, F: np.ndarray) -> "Verifier":
        self.refs_ = np.asarray(F, float)
        self.metric_, self.svm_ = self._fit(self.refs_)
        loo = []
        for i in range(len(self.refs_)):
            sub = np.delete(self.refs_, i, axis=0)
            loo.append(self._score(*self._fit(sub), self.refs_[i], sub))
        self.loo_ = np.array(loo)
        return self

    def score(self, q: np.ndarray) -> float:
        return self._score(self.metric_, self.svm_, q, self.refs_)

    def decide(self, q: np.ndarray, alpha: float = 1.0, zone: float = 1.0) -> dict:
        s = self.score(q); sd = float(np.std(self.loo_)) or 1e-6
        line = float(np.mean(self.loo_)) - alpha * sd
        verdict = "uncertain" if abs(s - line) <= zone * sd else ("genuine" if s > line else "forged")
        return dict(verdict=verdict, score=s, line=line, zone=(line - zone * sd, line + zone * sd),
                    consistency=self.metric_.consistency_)
