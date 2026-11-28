% Copyright (c) 2026 OGCZ. All rights reserved. See LICENSE.
function S = r1waw_writer_cov(Y, P)
%R1WAW_WRITER_COV  Covariance (1/n) of one writer's standardised, writer-centred pairs.
  Z = (P - mean(P, 1)) ./ Y.scale;
  S = (Z' * Z) / size(Z, 1);
end
