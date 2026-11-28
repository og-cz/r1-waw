% Copyright (c) 2026 OGCZ. All rights reserved. See LICENSE.
function Y = r1waw_yardstick(pairs)
%R1WAW_YARDSTICK  Phase A of R1-WAW: writer-independent constants from a development population.
%   Y = r1waw_yardstick(pairs)
%   pairs : cell array, one (n_w x d) matrix of genuine pair descriptors per development writer
%   Y     : struct with fields
%           scale   (1 x d)  std of each value over all pairs (+1e-9), population std (1/N)
%           cov     (d x d)  pooled within-writer covariance of the standardised pairs
%           eigvals (d x 1)  its eigenvalues, descending (eigvals(1) = lambda_1)
%           eigvecs (d x d)  eigenvectors as columns (eigvecs(:,1) = u_1)
%           W       (d x d)  cov^(-1/2), the shared whitening G
%   Works in MATLAB and GNU Octave (no toolbox).
  EPS = 1e-9;
  allP = vertcat(pairs{:});
  Y.scale = std(allP, 1, 1) + EPS;
  d = size(allP, 2);
  C = zeros(d);
  for i = 1:numel(pairs)
    C = C + r1waw_writer_cov(Y, pairs{i});
  end
  Y.cov = C / numel(pairs);
  [V, L] = eig((Y.cov + Y.cov') / 2);
  [ev, order] = sort(diag(L), 'descend');
  Y.eigvals = ev;
  Y.eigvecs = V(:, order);
  Y.W = r1waw_inv_sqrt(Y.cov);
end
