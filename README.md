# R1-WAW — Rank-One Writer-Adaptive Whitening

A writer-adaptive similarity metric for offline handwritten signature verification.

R1-WAW is a new kind of metric for writer-dependent verification. It learns how consistent each person is from that
person's genuine signatures alone, and it shapes the measure of similarity around that one trait. No forgeries are needed
at enrolment.

To the authors' knowledge, no earlier published method for offline signature verification adapts a shared similarity
metric to each writer in this way. Earlier methods do one of two things:
- they judge every writer with the same measure, or
- they estimate a full per-writer model, which twelve signatures cannot support.

R1-WAW sits between the two.

Implemented in **Python**, **MATLAB / GNU Octave**, **Julia** and **R**. All four pass the same numerical fixtures.

## The idea

Some people sign almost identically every time, and others vary a lot. A verifier that uses one fixed yardstick for
everybody is too lenient with a steady writer and too strict with a variable one.

R1-WAW keeps the yardstick that all writers share and changes it in only one place: the main direction along which
genuine signatures differ. Along that direction, each writer's yardstick is stretched or tightened to fit how consistent
that writer is. Every other direction stays shared.

- **One number per writer.** The adaptation needs only one value estimated from the writer's enrolment signatures.
  A full per-writer model would need 153.
- **Genuine signatures only.** Twelve genuine signatures are enough to enrol a writer.
- **Shared knowledge stays shared.** What is learned from the development population is fixed once and reused for
  every writer.
- **Safe default.** A writer of average consistency gets exactly the shared metric.

## Setting

- Enrolment: `n = 12` genuine signatures of one writer. No forgeries are used.
- Each signature becomes a fixed-length feature vector `h`. The reference study uses HOG with 2,916 values.
- A pair of signatures becomes `d = 17` dissimilarity values `φ(h_a, h_b)`. Twelve references give 66 enrolment
  pairs.
- A one-class SVM learns where the writer's genuine pairs lie. A questioned signature `q` is paired with each of the
  12 references, and its score is the mean decision value over those 12 pairs.

A shared whitening `G: x ↦ (x ⊘ s) Σ_pool^(−1/2)` treats every writer as equally variable. Estimating a full writer
covariance instead would need `d(d+1)/2 = 153` numbers from 12 signatures, which is unreliable. R1-WAW adapts one
direction only.

## The math

**Phase A: once, on development writers (genuine pairs only)**

```
s_j      = std of value j over all development pairs  (+ ε)
z        = (p − p̄_w) ⊘ s                               writer-centred, standardised pair
Σ_w      = (1/n_w) Σ z zᵀ
Σ_pool   = mean_w Σ_w = U Λ Uᵀ,   λ₁ ≥ … ≥ λ_d          frozen: s, Σ_pool, λ₁, u₁
```

**Phase B: per writer, from its 66 enrolment pairs**

```
σ_w²   = u₁ᵀ Σ_w u₁                       the writer's variance along u₁
c_w    = σ_w / √λ₁                        consistency (< 1: steadier than the average writer)
σ_eff² = λ₁ c_w^(2β)                      adaptation strength β = 3
W      = Σ_pool^(−1/2) + λ₁^(−1/2) (c_w^(−β) − 1) u₁u₁ᵀ
x  ↦   ((x − p̄_w) ⊘ s) W
```

The closed form is exact: `W = (Σ_pool + (σ_eff² − λ₁) u₁u₁ᵀ)^(−1/2)`, because `u₁` is an eigenvector of `Σ_pool`.
Along `u₁` the scale is `1/(√λ₁ c_w^β)`. A steady writer's departures along `u₁` are amplified, and a variable
writer's are damped. Every other direction keeps the pooled scale.

**Scoring**

```
OCSVM: ν = 0.10, RBF,  γ = k / (d · Var X),  kernel sharpness k = 3
s(q)  = (1/12) Σ_k f( transform(φ(h_q, h_k)) )
```

### Properties

1. `W Σ_eff W = I`, where `Σ_eff = Σ_pool + (σ_eff² − λ₁) u₁u₁ᵀ`.
2. If `c_w = 1` or `β = 0`, then `W = Σ_pool^(−1/2)` (`G`).
3. There is one estimated number per writer (`σ_w`). β and k are fixed constants, not per writer.

In the reference data, `u₁` carries 58% of the pooled within-writer variance and loads almost equally on the 17
values: it is the writer's overall dissimilarity level. `c_w` ranges from 0.30 to 3.09 across the development
writers (median 0.74).

### Settings

β = 3 and k = 3 were chosen on the 142 development writers among 56 combinations (k from 1 to 8, β from 0 to 3).
A split-half check picked the same pair on both halves. The narrower kernel (k = 3) also helps the shared metric
`G`, so the table below reports `G` at the same kernel, to isolate the writer-adaptive step.

## Pipeline

```
signature image ─► features (HOG) ─► pairwise dissimilarity ─► R1-WAW ─► one-class SVM ─► genuine / forged / uncertain
                                                                 ▲
                                          writer's 12 genuine signatures (enrolment)
```

A questioned signature is compared with each of the writer's 12 reference signatures. The verdict comes from a
per-writer decision line with an *uncertain* zone, so that borderline cases go to a human examiner.

## Results (reference study: CEDAR and BHSig260 Bengali / Hindi)

Each figure is the mean per-writer AUC for genuine signatures against skilled forgeries.

| Model | Development (142 writers) | Held-back (132 writers) |
|---|---|---|
| Shared metric, default kernel | 0.8309 | 0.8338 |
| Shared metric, same kernel as R1-WAW | 0.8497 | 0.8550 |
| **R1-WAW** | **0.8581** | **0.8602** |

Comparisons on the 132 held-back writers (Wilcoxon signed-rank, Holm-corrected):

| Comparison | Δ AUC | 95% CI | p |
|---|---|---|---|
| R1-WAW vs shared metric, default kernel | +0.0264 | +0.0132 to +0.0417 | 0.003 |
| R1-WAW vs shared metric, same kernel | +0.0052 | +0.0005 to +0.0100 | 0.051 |

- On Hindi, the writer-adaptive step alone adds +0.0117 (p = 0.006).
- The settings were chosen on the development writers. The held-back writers were scored afterwards.

**Operating rule.** On the 132 held-back writers, at the standard setting:
- 5.8% of genuine signatures are judged forged.
- 13.3% of forgeries are judged genuine.
- 37.6% of signatures are referred to an examiner as uncertain.

## Usage

**Python** (`pip install .[verifier]`)

```python
from r1waw import PooledYardstick, R1WAW, enrolment_pairs
from r1waw.verifier import Verifier

yard = PooledYardstick().fit([enrolment_pairs(F) for F in dev_features])   # once, on development writers
v = Verifier(yard).enrol(F_w)                                              # F_w: 12 genuine feature vectors
v.decide(h_q)          # {"verdict": "genuine" | "forged" | "uncertain", "score": ..., "consistency": ...}
```

**MATLAB / GNU Octave**

```matlab
Y = r1waw_yardstick(devPairs);  M = r1waw_fit(Y, P);  X = r1waw_transform(Y, M, Q);
```

**Julia**

```julia
include("julia/R1WAW.jl"); using .R1WAW
Y = yardstick(dev_pairs); M = R1WAW.fit(Y, P); X = transform(Y, M, Q)
```

**R**

```r
source("r/r1waw.R")
Y <- r1waw_yardstick(dev_pairs); M <- r1waw_fit(Y, P); X <- r1waw_transform(Y, M, Q)
```

The one-class SVM and the decision line are in the Python package. In the other languages, pass `X` to your own
one-class SVM library. Feature extraction is not included.

## Tests

```
pytest                                 # Python
octave --no-gui matlab/test_r1waw.m    # MATLAB / Octave
julia julia/test_r1waw.jl              # Julia
Rscript r/test_r1waw.R                 # R
```
