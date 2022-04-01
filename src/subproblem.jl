#####
##### generic subproblem interface
#####

"""
$(TYPEDEF)

A *linear model* for system of nonlinear equations, relative to the origin.

The L2 norm induces the objective function

```math
m(p) = p' J' r + 1/2 p' J' J p
```
approximating some `f(x) = \\| r(x) \\|^2_2` as

```math
1/2 \\| f(x + p) \\|_2^2 \approx 1/2 \\| r + J p \\|_2^2 = \\| r \\|_2^2 + p'J'r + 1/2 p' J' J p
```
"""
struct ResidualModel{TR,TJ}
    r::TR
    J::TJ
    function ResidualModel(r::TR, J::TJ) where {TR,TJ}
        @argcheck all(isfinite, r) "Non-finite residuals $(r)."
        @argcheck all(isfinite, J) "Non-finite Jacobian $(J)."
        n = length(r)
        @argcheck size(J) ≡ (n, n) "Non-conformable residual and Jacobian."
        new{TR,TJ}(r, J)
    end
end

"""
$(TYPEDEF)

A *quadratic model* for a minimization problem, relative to the origin, with the objective
function

```math
m(p) = p' g + 1/2 g' A g
```
"""
struct MinimizationModel{TG,TA}
    g::TG
    A::TA
    # FIXME constructor enforces symmetry of `A`, document, but do we actually rely on it?
end

"""
$(SIGNATURES)

Convert a model for a residual to a minimization model using the L2 norm.
"""
function MinimizationModel(model::ResidualModel)
    @unpack r, J = model
    MinimizationModel(J' * r, Symmetric(J' * J))
end

"""
$(SIGNATURES)

Find the Cauchy point (the minimum of the linear part, subject to ``\\| p \\| ≤ Δ``) of the problem.

Return three values:

1. the Cauchy point vector `pC`,

2. the (Euclidean) *norm* of `pC`

3. a boolean indicating whether the constraint was binding.

Caller guarantees non-zero gradient.
"""
function cauchy_point(Δ::Real, model::ResidualModel)
    @unpack r, J = model
    g = J' * r
    q = g' * (J' * J) * g
    g_norm = norm(g, 2)
    @argcheck g_norm > 0
    τ = if q ≤ 0              # practically 0 (semi-definite form) but allow for float error
        one(q)
    else
        min(one(q), g_norm^3 / (Δ * q))
    end
    (-τ * Δ / g_norm) .* g, τ * Δ, τ ≥ 1
end

"""
$(SIGNATURES)

Finds `\tau` such that

```math
\\| p_C + τ D \\|_2 = Δ
```

`pC_norm` is `\\| p_C \\|_2^2`, and can be provided when available.

Caller guarantees that

1. `pC_norm ≤ Δ`,

2. `norm(pC + D, 2) ≥ Δ`.

3. `dot(pC, D) ≥ 0`.

These ensure a meaningful solution `0 ≤ τ ≤ 1`. None of them are checked explicitly.
"""
function dogleg_boundary(Δ, D, pC, pC_norm = norm(pC, 2))
    a = sum(abs2, D)
    b = 2 * dot(D, pC)
    c = abs2(pC_norm) - abs2(Δ)
    (-b + √(abs2(b) - 4 * a * c)) / (2 * a)
end

"""
$(SIGNATURES)

Calculate the unconstrained optimum of the model, return its norm as the second value.

When the second value is *infinite*, the unconstrained optimum should not be used as this
indicates a singular problem.
"""
function unconstrained_optimum(model::ResidualModel)
    @unpack r, J = model
    F, is_valid_F = _factorize(J)
    if is_valid_F
        pU = -(F \ r)
        pU, norm(pU, 2)
    else
        ∞ = oftype(one(eltype(r)) / one(eltype(F)), Inf)
        fill(∞, length(r)), ∞
    end
end

"""
`$(FUNCTIONNAME)(method, Δ, model)`

Optimize the `model` with the given constraint `Δ` using `method`. Returns the following
values:

- the optimum,
- the norm of the optimum,
- a boolean indicating whether the constraint binds.
"""
function solve_model end
