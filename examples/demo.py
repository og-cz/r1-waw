# Copyright (c) 2026 OGCZ. All rights reserved. See LICENSE.
"""Synthetic illustration of the mechanism (not evidence): pair vectors generated directly.

Writers share one pooled covariance except along its leading direction u_1, where writer w has variance c_w^2 * lambda_1
(c_w log-normal: steady ... variable writers). Forgery pairs are shifted along u_1 (larger dissimilarity) plus noise.
Compares G with R1-WAW at beta = 1, 2, 3. Needs scikit-learn.
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM

from r1waw import R1WAW, PooledYardstick
from r1waw.verifier import kernel_gamma

rng = np.random.default_rng(0)
D, N_DEV, N_TEST, N_ENROL_PAIRS, N_REF = 17, 140, 140, 66, 12
Q, _ = np.linalg.qr(rng.normal(size=(D, D)))
lam = 0.8 ** np.arange(D) * 4.0
mu_pop = rng.normal(size=D) * 5


def writer():
    c = rng.lognormal(0.0, 0.5)
    cov = Q @ np.diag(np.r_[lam[0] * c ** 2, lam[1:]]) @ Q.T
    mu = mu_pop + rng.normal(size=D)
    shift = 1.5 * np.sqrt(lam[0]) * Q[:, 0] + Q @ (np.sqrt(lam) * rng.normal(size=D) * 0.3)
    enrol = rng.multivariate_normal(mu, cov, N_ENROL_PAIRS)
    gen = [rng.multivariate_normal(mu, cov, N_REF) for _ in range(12)]
    forg = [rng.multivariate_normal(mu + shift, cov, N_REF) for _ in range(12)]
    return enrol, gen, forg


yard = PooledYardstick().fit([writer()[0] for _ in range(N_DEV)])              # Phase A: development writers only
test = [writer() for _ in range(N_TEST)]
# Only the adaptation strength is illustrated; the kernel sharpness was chosen on real development writers
# and is a property of the data, so it is not demonstrated on synthetic pairs.
for name, beta, sharp in [("G (shared)", 0.0, 1.0), ("beta 1", 1.0, 1.0), ("beta 2", 2.0, 1.0),
                          ("beta 3", 3.0, 1.0)]:
    aucs = []
    for enrol, gen, forg in test:
        m = R1WAW(yard, beta).fit(enrol); X = m.transform(enrol)                 # Phase B: one number per writer
        svm = OneClassSVM(kernel="rbf", nu=0.10, gamma=kernel_gamma(X, sharp)).fit(X)
        s = [svm.decision_function(m.transform(q)).mean() for q in gen + forg]
        aucs.append(roc_auc_score(np.r_[np.ones(len(gen)), np.zeros(len(forg))], s))
    print(f"{name:22s} mean per-writer AUC {np.mean(aucs):.4f}")
