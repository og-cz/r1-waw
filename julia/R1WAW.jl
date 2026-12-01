# Copyright (c) 2026 OGCZ. All rights reserved. See LICENSE.
"""
    R1WAW

Rank-One Writer-Adaptive Whitening. Standard library only (LinearAlgebra, Statistics).

    Y = yardstick(pairs)              # Phase A: pairs = Vector of (n_w × d) matrices, development writers
    M = fit(Y, P; β = 3.0)            # Phase B: P = (n × d) genuine pairs of one writer
    X = transform(Y, M, Q)            # pairs in the writer's space
β: adaptation strength (default 3; β = 0 gives the shared whitening G).
"""
module R1WAW

using LinearAlgebra, Statistics

export Yardstick, Metric, yardstick, fit, transform, inv_sqrt, writer_cov, pair_descriptor, enrolment_pairs

const EPS = 1e-9
const EV_FLOOR = 1e-8

struct Yardstick
    scale::Vector{Float64}      # std of each value over all pairs (+EPS), population std
    cov::Matrix{Float64}        # pooled within-writer covariance of standardised pairs
    eigvals::Vector{Float64}    # descending; eigvals[1] = λ₁
    eigvecs::Matrix{Float64}    # columns; eigvecs[:, 1] = u₁
    W::Matrix{Float64}          # cov^(-1/2): shared whitening G
end

struct Metric
    mean::Vector{Float64}
    sigma2::Float64             # u₁ᵀ Σ_w u₁
    consistency::Float64        # c_w = sqrt(sigma2 / λ₁)
    W::Matrix{Float64}
    β::Float64
end

function inv_sqrt(S::AbstractMatrix)
    F = eigen(Symmetric((S + S') / 2))
    return F.vectors * Diagonal(max.(F.values, EV_FLOOR) .^ -0.5) * F.vectors'
end

writer_cov(scale, P) = (Z = (P .- mean(P, dims = 1)) ./ scale'; Z' * Z / size(Z, 1))

function yardstick(pairs::AbstractVector)
    allP = reduce(vcat, pairs)
    scale = vec(std(allP, dims = 1, corrected = false)) .+ EPS
    C = sum(writer_cov(scale, P) for P in pairs) / length(pairs)
    F = eigen(Symmetric((C + C') / 2)); order = sortperm(F.values, rev = true)
    return Yardstick(scale, C, F.values[order], F.vectors[:, order], inv_sqrt(C))
end

function fit(Y::Yardstick, P::AbstractMatrix; β::Real = 3.0)
    u1, λ1 = Y.eigvecs[:, 1], Y.eigvals[1]
    μ = vec(mean(P, dims = 1))
    z = ((P .- μ') ./ Y.scale') * u1
    σ2 = mean(z .^ 2)
    s_eff = λ1 * (max(σ2, EV_FLOOR) / λ1)^β
    W = Y.W + (s_eff^-0.5 - λ1^-0.5) * (u1 * u1')
    return Metric(μ, σ2, sqrt(σ2 / λ1), W, Float64(β))
end

transform(Y::Yardstick, M::Metric, Q::AbstractMatrix) = ((Q .- M.mean') ./ Y.scale') * M.W

function pct_linear(x, p)
    s = sort(vec(x)); h = (length(s) - 1) * p / 100; lo = floor(Int, h); hi = min(lo + 1, length(s) - 1)
    return s[lo + 1] + (h - lo) * (s[hi + 1] - s[lo + 1])
end

"Symmetric 17-value dissimilarity of two feature vectors (same order as the Python package)."
function pair_descriptor(a::AbstractVector, b::AbstractVector)
    a = Float64.(a); b = Float64.(b); d = abs.(a .- b)
    ac, bc = a .- mean(a), b .- mean(b)
    sd(x) = std(x, corrected = false)
    return [mean(d), sqrt(sum(d .^ 2)), maximum(d),
            1 - dot(a, b) / (norm(a) * norm(b) + EPS),
            1 - dot(ac, bc) / (norm(ac) * norm(bc) + EPS),
            sum(d ./ (abs.(a) .+ abs.(b) .+ EPS)),
            sum(d) / (sum(abs.(a .+ b)) + EPS),
            sd(d), median(d), pct_linear(d, 90),
            abs(mean(a) - mean(b)), abs(sd(a) - sd(b)),
            abs(maximum(a) - maximum(b)), abs(sum(a .^ 2) - sum(b .^ 2)),
            sum(d .^ 2 ./ (abs.(a) .+ abs.(b) .+ EPS)),
            1 - sum(min.(abs.(a), abs.(b))) / (max(sum(abs.(a)), sum(abs.(b))) + EPS),
            abs(sum(a) - sum(b))]
end

"All unordered pairs of the reference vectors (rows of F): (n(n-1)/2 × 17)."
enrolment_pairs(F::AbstractMatrix) =
    reduce(vcat, [pair_descriptor(F[i, :], F[j, :])' for i in 1:size(F, 1) for j in i+1:size(F, 1)])

end # module
