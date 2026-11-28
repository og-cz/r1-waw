% Copyright (c) 2026 OGCZ. All rights reserved. See LICENSE.
function X = r1waw_transform(Y, M, Q)
%R1WAW_TRANSFORM  Pairs Q (n x d) measured in the writer's space: ((Q - mean) ./ scale) * W.
  X = ((Q - M.mean) ./ Y.scale) * M.W;
end
