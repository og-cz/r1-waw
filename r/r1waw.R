# Copyright (c) 2026 OGCZ. All rights reserved. See LICENSE.
# Rank-One Writer-Adaptive Whitening (R1-WAW) in base R.
#   Y <- r1waw_yardstick(pairs)          Phase A: list of (n_w x d) matrices, development writers
#   M <- r1waw_fit(Y, P, beta = 3)       Phase B: (n x d) genuine pairs of one writer
#   X <- r1waw_transform(Y, M, Q)        pairs in the writer's space
# beta: adaptation strength (default 3; beta = 0 gives the shared whitening G).

R1WAW_EPS <- 1e-9
R1WAW_EV_FLOOR <- 1e-8

r1waw_inv_sqrt <- function(S) {
  e <- eigen((S + t(S)) / 2, symmetric = TRUE)
  e$vectors %*% diag(pmax(e$values, R1WAW_EV_FLOOR)^-0.5, nrow = length(e$values)) %*% t(e$vectors)
}

pop_sd <- function(x) sqrt(mean((x - mean(x))^2))

r1waw_writer_cov <- function(scale, P) {
  Z <- sweep(sweep(P, 2, colMeans(P)), 2, scale, "/")
  crossprod(Z) / nrow(Z)
}

r1waw_yardstick <- function(pairs) {
  allP <- do.call(rbind, pairs)
  scale <- apply(allP, 2, pop_sd) + R1WAW_EPS
  C <- Reduce(`+`, lapply(pairs, function(P) r1waw_writer_cov(scale, P))) / length(pairs)
  e <- eigen((C + t(C)) / 2, symmetric = TRUE)          # values already in decreasing order
  list(scale = scale, cov = C, eigvals = e$values, eigvecs = e$vectors, W = r1waw_inv_sqrt(C))
}

r1waw_fit <- function(Y, P, beta = 3) {
  u1 <- Y$eigvecs[, 1]; lam1 <- Y$eigvals[1]
  mu <- colMeans(P)
  z <- sweep(sweep(P, 2, mu), 2, Y$scale, "/") %*% u1
  sigma2 <- mean(z^2)
  s_eff <- lam1 * (max(sigma2, R1WAW_EV_FLOOR) / lam1)^beta
  W <- Y$W + (s_eff^-0.5 - lam1^-0.5) * tcrossprod(u1)
  list(mean = mu, sigma2 = sigma2, consistency = sqrt(sigma2 / lam1), W = W, beta = beta)
}

r1waw_transform <- function(Y, M, Q) sweep(sweep(Q, 2, M$mean), 2, Y$scale, "/") %*% M$W

r1waw_pair_descriptor <- function(a, b) {
  eps <- R1WAW_EPS; a <- as.numeric(a); b <- as.numeric(b); d <- abs(a - b)
  ac <- a - mean(a); bc <- b - mean(b); nrm <- function(x) sqrt(sum(x^2))
  c(mean(d), sqrt(sum(d^2)), max(d),
    1 - sum(a * b) / (nrm(a) * nrm(b) + eps),
    1 - sum(ac * bc) / (nrm(ac) * nrm(bc) + eps),
    sum(d / (abs(a) + abs(b) + eps)),
    sum(d) / (sum(abs(a + b)) + eps),
    pop_sd(d), median(d), unname(quantile(d, 0.9, type = 7)),
    abs(mean(a) - mean(b)), abs(pop_sd(a) - pop_sd(b)),
    abs(max(a) - max(b)), abs(sum(a^2) - sum(b^2)),
    sum(d^2 / (abs(a) + abs(b) + eps)),
    1 - sum(pmin(abs(a), abs(b))) / (max(sum(abs(a)), sum(abs(b))) + eps),
    abs(sum(a) - sum(b)))
}

r1waw_enrolment_pairs <- function(F) {
  idx <- utils::combn(nrow(F), 2)
  t(apply(idx, 2, function(ij) r1waw_pair_descriptor(F[ij[1], ], F[ij[2], ])))
}
