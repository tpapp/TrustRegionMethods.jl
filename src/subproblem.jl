#####
##### local subproblem interface and utility functions
#####

"""
$(TYPEDEF)

A *quadratic model* for a minimization problem, relative to the origin, with the objective
function

```math
m(p) = f + p' g + 1/2 p' B p
```
"""
struct LocalModel{TF<:Real,TG<:AbstractVector,TB<:AbstractMatrix}
    f::TF
    g::TG
    B::TB
    function LocalModel(f::TF, g::TG, B::TB) where {TF,TG,TB}
        n = length(g)
        @argcheck size(B) == (n, n) DimensionMismatch
        @argcheck isfinite(f)
        @argcheck all(isfinite, g)
        @argcheck all(isfinite, B)
        new{TF,TG,TB}(f, g, B)
    end
    # FIXME we could store information about the definiteness of B, ie if it is known, and
    # what it is
end

"""
$(SIGNATURES)

Evaluate the model at the origin (``0`3`).
"""
value_at_origin(model::LocalModel) = model.f

"""
$(SIGNATURES)

Convert a residual from a system of equations (in a rootfinding problem) to an objective for
minimization.
"""
residual_minimand(r) = sum(abs2, r)

"""
$(SIGNATURES)

Let `r` and `J` be residuals and their Jacobian at some point. We construct a local model as

```math
m(p) = 1/2 \\| J p - r \\|^2_2 = \\| r \\|^2_2 + p' J ' r + 1/2 p' J' J p
```
"""
function local_residual_model(r::AbstractVector, J::AbstractMatrix)
    # FIXME B always p.s.d, cf note above
    m = residual_minimand(r)
    @argcheck all(isfinite, m) "Cannot build a model for non-finite residuals."
    LocalModel(m, J' * r, SELF' * J)
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
function cauchy_point(Δ::Real, model::LocalModel)
    (; g, B) = model
    g_norm = norm(g, 2)
    q = ellipsoidal_norm(g, B)
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

Calculate the unconstrained optimizer of the model, return its norm as the second value.

When the second value is *infinite*, the unconstrained optimizer should not be used as this
indicates a singular problem.
"""
function unconstrained_optimizer(model::LocalModel)
    (; g, B) = model
    F, is_valid_F = _factorize(B)
    if is_valid_F
        pU = -(F \ g)
        pU, norm(pU, 2)
    else
        ∞ = oftype(one(eltype(g)) / one(eltype(F)), Inf)
        fill(∞, length(g)), ∞
    end
end

"""
`$(FUNCTIONNAME)(method, Δ, model)`

Optimize the `model` with the given constraint `Δ` using `method`. Returns the following
values:

- the optimizer (a vector),
- the norm of the optimizer (a scalar),
- a boolean indicating whether the constraint binds.
"""
function solve_model end

"""
$(SIGNATURES)

Reduction of model objective at `p`, compared to the origin.
"""
function calculate_model_reduction(model::LocalModel, p)
    (; g, B) = model
    - (dot(p, g) + ellipsoidal_norm(p, B) / 2)
end
