# Copyright (c) 2026 OGCZ. All rights reserved. See LICENSE.
from pathlib import Path

import numpy as np
import pytest

from r1waw import PooledYardstick, R1WAW, enrolment_pairs, inv_sqrt, pair_descriptor, r1waw_metric

FIX = Path(__file__).parent / "fixtures"


def population(n_writers=40, n_pairs=66, d=17, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, d))
    base = A @ A.T / d + np.eye(d) * 0.1
    return [rng.multivariate_normal(rng.normal(size=d), base * rng.lognormal(0, 0.4), size=n_pairs)
            for _ in range(n_writers)]


@pytest.fixture(scope="module")
def fitted():
    pairs = population()
    return pairs, PooledYardstick().fit(pairs)


def test_closed_form_equals_eigenvalue_replacement(fitted):
    pairs, y = fitted
    for beta in (0.5, 1.0, 2.0, 3.0):
        m = R1WAW(y, beta).fit(pairs[3])
        lam = y.eigvals_.copy(); lam[0] = lam[0] * (m.sigma2_ / lam[0]) ** beta
        assert np.allclose(m.W_, inv_sqrt(y.eigvecs_ @ np.diag(lam) @ y.eigvecs_.T), atol=1e-8)


def test_whitens_the_adapted_covariance(fitted):
    pairs, y = fitted
    m = R1WAW(y, beta=1.0).fit(pairs[5])
    u1, lam1 = y.eigvecs_[:, 0], y.eigvals_[0]
    S = y.cov_ + (m.sigma2_ - lam1) * np.outer(u1, u1)
    assert np.allclose(m.W_ @ S @ m.W_, np.eye(len(S)), atol=1e-6)


def test_reduces_to_shared_whitening(fitted):
    pairs, y = fitted
    assert np.allclose(R1WAW(y, beta=0.0).fit(pairs[0]).W_, y.W_)
    for beta in (1.0, 2.0, 3.0):
        assert np.allclose(r1waw_metric(y, y.eigvals_[0], beta), y.W_)


def test_weight_along_u1(fitted):
    pairs, y = fitted
    u1 = y.eigvecs_[:, 0]
    for beta in (1.0, 2.0, 3.0):
        m = R1WAW(y, beta).fit(pairs[2])
        assert np.isclose(u1 @ m.W_ @ u1, 1 / (np.sqrt(y.eigvals_[0]) * m.consistency_ ** beta))


def test_sigma2_is_variance_along_u1(fitted):
    pairs, y = fitted
    m = R1WAW(y).fit(pairs[7])
    u1 = y.eigvecs_[:, 0]
    assert np.isclose(m.sigma2_, u1 @ y.writer_cov(pairs[7]) @ u1)
    assert np.isclose(m.consistency_, np.sqrt(m.sigma2_ / y.eigvals_[0]))


def test_descriptor():
    rng = np.random.default_rng(1)
    a, b = rng.random(2916), rng.random(2916)
    assert np.allclose(pair_descriptor(a, b), pair_descriptor(b, a))
    assert np.allclose(pair_descriptor(a, a), 0, atol=1e-6)
    assert enrolment_pairs(rng.random((12, 50))).shape == (66, 17)


def test_fixtures_match():
    """The CSV fixtures used by the MATLAB/Octave, Julia and R tests agree with this implementation."""
    load = lambda n: np.loadtxt(FIX / n, delimiter=",")
    data = load("pairs.csv"); pairs = [data[data[:, 0] == w, 1:] for w in np.unique(data[:, 0])]
    y = PooledYardstick().fit(pairs)
    assert np.allclose(y.scale_, load("scale.csv")) and np.allclose(y.W_, load("W_pool.csv"))
    for beta in (1, 2, 3):
        assert np.allclose(R1WAW(y, beta).fit(pairs[0]).W_, load(f"W_beta{beta}.csv"))
    ab = load("desc_ab.csv")
    assert np.allclose(pair_descriptor(ab[0], ab[1]), load("desc_expected.csv"))
