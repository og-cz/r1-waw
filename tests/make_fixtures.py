# Copyright (c) 2026 OGCZ. All rights reserved. See LICENSE.
"""Writes the shared test fixtures (CSV) that every language implementation is checked against.

    python tests/make_fixtures.py        (run from the repository root)
"""
from pathlib import Path

import numpy as np

from r1waw import PooledYardstick, R1WAW, pair_descriptor

OUT = Path(__file__).parent / "fixtures"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(20261027)
d, n_writers, n_pairs = 17, 20, 66
A = rng.normal(size=(d, d)); base = A @ A.T / d + 0.1 * np.eye(d)
pairs = [rng.multivariate_normal(rng.normal(size=d), base * rng.lognormal(0, 0.4), n_pairs) for _ in range(n_writers)]
save = lambda name, x: np.savetxt(OUT / name, np.atleast_2d(x), delimiter=",", fmt="%.17g")

save("pairs.csv", np.column_stack([np.repeat(np.arange(n_writers), n_pairs), np.vstack(pairs)]))   # col 0 = writer
y = PooledYardstick().fit(pairs)
save("scale.csv", y.scale_); save("eigvals.csv", y.eigvals_); save("W_pool.csv", y.W_)
for beta in (1, 2, 3):
    m = R1WAW(y, beta).fit(pairs[0])
    save(f"W_beta{beta}.csv", m.W_)
save("sigma2_consistency.csv", [m.sigma2_, m.consistency_])
a, b = rng.random(50), rng.random(50)
save("desc_ab.csv", np.vstack([a, b])); save("desc_expected.csv", pair_descriptor(a, b))
print("fixtures written to", OUT)
