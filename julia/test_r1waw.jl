# Copyright (c) 2026 OGCZ. All rights reserved. See LICENSE.
# Checks the Julia implementation against the shared fixtures. Run from the repository root:
#     julia julia/test_r1waw.jl
using DelimitedFiles, LinearAlgebra
include(joinpath(@__DIR__, "R1WAW.jl")); using .R1WAW
fx(n) = readdlm(joinpath(@__DIR__, "..", "tests", "fixtures", n), ',', Float64)
data = fx("pairs.csv")
pairs = [data[data[:, 1] .== w, 2:end] for w in sort(unique(data[:, 1]))]
Y = yardstick(pairs)
close(a, b; tol = 1e-8) = maximum(abs.(vec(a) .- vec(b))) < tol
@assert close(Y.scale, fx("scale.csv"))
@assert close(Y.eigvals, fx("eigvals.csv"))
@assert close(Y.W, fx("W_pool.csv"))
local M
for β in 1:3
    global M = R1WAW.fit(Y, pairs[1]; β = β)
    @assert close(M.W, fx("W_beta$(β).csv"); tol = 1e-7)
end
sc = fx("sigma2_consistency.csv"); @assert close([M.sigma2, M.consistency], sc)
ab = fx("desc_ab.csv"); @assert close(pair_descriptor(ab[1, :], ab[2, :]), fx("desc_expected.csv"); tol = 1e-9)
@assert size(enrolment_pairs(rand(12, 30))) == (66, 17)
println("Julia: all checks passed")
