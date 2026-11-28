% Copyright (c) 2026 OGCZ. All rights reserved. See LICENSE.
function P = r1waw_enrolment_pairs(F)
%R1WAW_ENROLMENT_PAIRS  All unordered pairs of the n reference vectors (rows of F): (n(n-1)/2 x 17).
  n = size(F, 1); P = zeros(n * (n - 1) / 2, 17); r = 0;
  for i = 1:n - 1
    for j = i + 1:n
      r = r + 1; P(r, :) = r1waw_pair_descriptor(F(i, :), F(j, :));
    end
  end
end
