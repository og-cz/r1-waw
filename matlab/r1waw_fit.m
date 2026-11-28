% Copyright (c) 2026 OGCZ. All rights reserved. See LICENSE.
function M = r1waw_fit(Y, P, beta)
%R1WAW_FIT  Phase B of R1-WAW: the writer's metric from its genuine enrolment pairs only.
%   M = r1waw_fit(Y, P)        beta = 3 (default)
%   M = r1waw_fit(Y, P, beta)  adaptation strength; beta = 0 gives the shared whitening G
%   P : (n x d) genuine pair descriptors of one writer (66 x 17 for 12 references)
%   M : struct with fields mean (1 x d), sigma2, consistency (c_w), W (d x d), beta
%   Closed form:  W = Sigma_pool^(-1/2) + (s_eff^(-1/2) - lambda_1^(-1/2)) u_1 u_1',
%                 s_eff = lambda_1 * (sigma2 / lambda_1)^beta.
  if nargin < 3, beta = 3; end
  u1 = Y.eigvecs(:, 1); lam1 = Y.eigvals(1);
  M.beta = beta;
  M.mean = mean(P, 1);
  z = ((P - M.mean) ./ Y.scale) * u1;
  M.sigma2 = mean(z .^ 2);
  M.consistency = sqrt(M.sigma2 / lam1);
  s_eff = lam1 * (max(M.sigma2, 1e-8) / lam1) ^ beta;
  M.W = Y.W + (s_eff ^ -0.5 - lam1 ^ -0.5) * (u1 * u1');
end
