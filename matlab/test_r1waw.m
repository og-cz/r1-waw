% Copyright (c) 2026 OGCZ. All rights reserved. See LICENSE.
% Checks the MATLAB/Octave implementation against the shared fixtures (tests/fixtures).
% Run from the repository root:   octave --no-gui matlab/test_r1waw.m     or in MATLAB: run('matlab/test_r1waw.m')
here = fileparts(mfilename('fullpath')); addpath(here);
fx = fullfile(here, '..', 'tests', 'fixtures');
ld = @(n) csvread(fullfile(fx, n));
data = ld('pairs.csv'); ids = unique(data(:, 1)); pairs = cell(numel(ids), 1);
for i = 1:numel(ids), pairs{i} = data(data(:, 1) == ids(i), 2:end); end
Y = r1waw_yardstick(pairs);
tol = 1e-8; ok = true;
ok = ok && max(abs(Y.scale - ld('scale.csv'))) < tol;
ok = ok && max(abs(Y.eigvals' - ld('eigvals.csv'))) < tol;
ok = ok && max(max(abs(Y.W - ld('W_pool.csv')))) < tol;
for beta = 1:3
  M = r1waw_fit(Y, pairs{1}, beta);
  ok = ok && max(max(abs(M.W - ld(sprintf('W_beta%d.csv', beta))))) < 1e-7;
end
sc = ld('sigma2_consistency.csv'); ok = ok && abs(M.sigma2 - sc(1)) < tol && abs(M.consistency - sc(2)) < tol;
ab = ld('desc_ab.csv'); ok = ok && max(abs(r1waw_pair_descriptor(ab(1, :), ab(2, :)) - ld('desc_expected.csv'))) < 1e-9;
ok = ok && isequal(size(r1waw_enrolment_pairs(rand(12, 30))), [66 17]);
if ok, disp('MATLAB/Octave: all checks passed'); else, error('MATLAB/Octave: a check failed'); end
