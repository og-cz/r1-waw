# Copyright (c) 2026 OGCZ. All rights reserved. See LICENSE.
# Checks the R implementation against the shared fixtures. Run from the repository root:
#     Rscript r/test_r1waw.R
source(file.path("r", "r1waw.R"))
fx <- function(n) as.matrix(read.csv(file.path("tests", "fixtures", n), header = FALSE))
data <- fx("pairs.csv")
pairs <- lapply(sort(unique(data[, 1])), function(w) data[data[, 1] == w, -1, drop = FALSE])
Y <- r1waw_yardstick(pairs)
near <- function(a, b, tol = 1e-8) max(abs(as.numeric(a) - as.numeric(b))) < tol
stopifnot(near(Y$scale, fx("scale.csv")), near(Y$eigvals, fx("eigvals.csv")), near(Y$W, fx("W_pool.csv")))
for (beta in 1:3) {
  M <- r1waw_fit(Y, pairs[[1]], beta)
  stopifnot(near(M$W, fx(sprintf("W_beta%d.csv", beta)), 1e-7))
}
stopifnot(near(c(M$sigma2, M$consistency), fx("sigma2_consistency.csv")))
ab <- fx("desc_ab.csv")
stopifnot(near(r1waw_pair_descriptor(ab[1, ], ab[2, ]), fx("desc_expected.csv"), 1e-9))
stopifnot(all(dim(r1waw_enrolment_pairs(matrix(runif(360), 12, 30))) == c(66, 17)))
cat("R: all checks passed\n")
