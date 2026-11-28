% Copyright (c) 2026 OGCZ. All rights reserved. See LICENSE.
function W = r1waw_inv_sqrt(S)
%R1WAW_INV_SQRT  Symmetric inverse square root S^(-1/2), eigenvalues floored at 1e-8.
  [V, L] = eig((S + S') / 2);
  W = V * diag(max(diag(L), 1e-8) .^ -0.5) * V';
end
