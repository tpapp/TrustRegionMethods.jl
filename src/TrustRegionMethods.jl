module TrustRegionMethods

export trust_region_solver, TrustRegionResult, ForwardDiff_wrapper

using ArgCheck: @argcheck
import DiffResults
using DocStringExtensions: FIELDS, SIGNATURES, TYPEDEF
import ForwardDiff
using LinearAlgebra: dot, norm
using UnPack: @unpack

####
#### building blocks
####

"""
$(TYPEDEF)

A *linear model* for system of nonlinear equations, relative to the origin, as

```math
m(p) = p' J' r + 1/2 p' J' J p
```

approximating some `f(x) = \\| r(x) \\|^2_2` as

```math
1/2 \\| f(x + p) \\|_2^2 \approx 1/2 \\| r + J p \\|_2^2 = \\| r \\|_2^2 + p'J'r + 1/2 p' J' J p
```
"""
struct NonlinearModel{TR,TJ}
    r::TR
    J::TJ
    function NonlinearModel(r::TR, J::TJ) where {TR,TJ}
        @argcheck all(isfinite, r) "Non-finite residuals $(r)."
        @argcheck all(isfinite, J) "Non-finite residuals $(J)."
        n = length(r)
        @argcheck size(J) ≡ (n, n) "Non-conformable residual and Jacobian."
        new{TR,TJ}(r, J)
    end
end

"""
$(SIGNATURES)

Find the Cauchy point (the minimum of the linear part, subject to ``\\| p \\| ≤ Δ``) of the problem.

Return three values:

1. the Cauchy point vector `pC`,

2. the (Euclidean) *norm* of `pC`

3. a boolean indicating whether the constraint was binding.
"""
function cauchy_point(Δ::Real, model::NonlinearModel)
    @unpack r, J = model
    g = J' * r
    q = g' * (J' * J) * g
    g_norm = norm(g, 2)
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

Implementation of the *dogleg method*. Return the minimizer and a boolean indicating if the
constraint is binding.
"""
function dogleg(Δ, model::NonlinearModel)
    @unpack r, J = model
    pU = -(J \ r)               # unconstrained step
    pU_norm = norm(pU, 2)
    if pU_norm ≤ Δ
        pU, false
    else
        pC, pC_norm, on_boundary = cauchy_point(Δ, model)
        if on_boundary
            pC, true
        else
            D = pU .- pC
            τ = dogleg_boundary(Δ, D, pC, pC_norm)
            pC .+ D .* τ, true
        end
    end
end

####
#### trust region steps
####

struct TrustRegionParameters{T}
    η::T
    Δ̄::T
    function TrustRegionParameters(η::T, Δ̄::T) where {T <: Real}
        @argcheck 0 < η < 0.25
        @argcheck Δ̄ > 0
        new{T}(η, Δ̄)
    end
end

TrustRegionParameters(η, Δ̄) = TrustRegionParameterS(promote(η, Δ̄)...)

TrustRegionParameters(; η = 0.125, Δ̄ = Inf) = TrustRegionParameters(η, Δ̄)

"""
$(SIGNATURES)

Ratio between predicted (using `model`, at `p`) and actual reduction (taken from the
residual value `r′`). Will return an arbitrary negative number for infeasible coordinates.
"""
reduction_ratio(fx, p, ::Nothing) = -one(eltype(p))

function reduction_ratio(model::NonlinearModel, p, r′)
    @unpack r, J = model
    @argcheck all(isfinite, r) && all(isfinite, J) "residual or Jacobian are not finite"
    r2 = sum(abs2, r)
    (r2 - sum(abs2, r′)) / (r2 - sum(abs2, r .+ J * p))
end

function trust_region_step(parameters::TrustRegionParameters, f, Δ, x, fx)
    @unpack η, Δ̄ = parameters
    model = NonlinearModel(fx.residual, fx.Jacobian)
    p, on_boundary = dogleg(Δ, model)
    x′ = x .+ p
    fx′ = f(x′)
    ρ = reduction_ratio(model, p, fx′.residual)
    Δ′ =
        if ρ < 0.25
            norm(p, 2) / 4
        elseif ρ > 0.75 && on_boundary
            min(2 * Δ, Δ̄)
        else
            Δ
        end
    if ρ ≥ η
        Δ′, x′, fx′            # reasonable reduction, use new position
    else
        Δ′, x, fx              # very small reduction, try again from original
    end
end

struct NonlinearStoppingCriterion{T <: Real}
    residual_norm_tolerance::T
    function NonlinearStoppingCriterion(residual_norm_tolerance::T) where {T}
        @argcheck residual_norm_tolerance > 0
        new{T}(residual_norm_tolerance)
    end
end

function NonlinearStoppingCriterion(; residual_norm_tolerance = √eps())
    NonlinearStoppingCriterion(residual_norm_tolerance)
end

function check_stopping_criterion(nsc::NonlinearStoppingCriterion, fx)
    r_norm = norm(fx.residual, 2)
    converged = r_norm ≤ nsc.residual_norm_tolerance
    converged, (converged = converged , residual_norm = r_norm)
end

"""
$(TYPEDEF)

# Fields

$(FIELDS)
"""
struct TrustRegionResult{T,TX,TFX}
    "The final trust region radius."
    Δ::T
    "The last value (the root only when converged)."
    x::TX
    "`f(x)` at the last `x`."
    fx::TFX
    "The Euclidean norm of the residual at `x`."
    residual_norm::T
    "A boolean indicating convergence."
    converged::Bool
    "Number of iterations (≈ number of function evaluations)."
    iterations::Int
end

function Base.show(io::IO, ::MIME"text/plain", trr::TrustRegionResult)
    @unpack Δ, x, fx, residual_norm, converged, iterations = trr
    _sig(x) = round(x; sigdigits = 3)
    print(io, "Nonlinear solver using trust region method ")
    if converged
        printstyled(io, "converged"; color = :green)
    else
        printstyled(io, "didn't converge"; color = :red)
    end
    println(io, " after $(iterations) steps")
    print(io, "  with ")
    printstyled(io, "‖x‖₂ = $(_sig(residual_norm)), Δ = $(_sig(Δ))\n"; color = :blue)
    println(io, "  x = ", _sig.(x))
    println(io, "  r = ", _sig.(fx.residual))
end

"""
$(SIGNATURES)

Solve `f ≈ 0` using trust region methods, starting from `x`.

`f` is a callable (function) that

1. takes vectors real numbers of the same length as `x`,

2. returns a *either*

    a. `nothing`, if the objective cannot be evaluated,

    b. a value with properties `residual` and `Jacobian`, containing a vector and a
    conformable matrix, with finite values. A `NamedTuple` can be used for this, but any
    structure with these properties will be accepted. Importantly, this is treated as a
    single object and can be used to return extra information (a “payload”), which will be
    ignored by this function.

Returns a [`TrustRegionResult`](@ref) object.
"""
function trust_region_solver(f, x;
                             parameters = TrustRegionParameters(),
                             stopping_criterion = NonlinearStoppingCriterion(),
                             maximum_iterations = 500,
                             Δ = 1.0)
    fx = f(x)
    iterations = 1
    while true
        Δ, x, fx = trust_region_step(parameters, f, Δ, x, fx)
        iterations += 1
        do_stop, convergence_statistics = check_stopping_criterion(stopping_criterion, fx)
        reached_max_iter = iterations ≥ maximum_iterations
        if do_stop || reached_max_iter
            @unpack converged, residual_norm = convergence_statistics
            return TrustRegionResult(Δ, x, fx, residual_norm, converged, iterations)
        end
    end
end

####
#### AD shims
####

###
### ForwardDiff
###

"""
$(TYPEDEF)

A buffer for wrapping a (nonlinear) mapping `f` to calculate the Jacobian with
`ForwardDiff.jacobian`.
"""
struct ForwardDiffBuffer{TF,TR,TC}
    "A function that maps vectors to residual vectors."
    f::TF
    "`DiffResults` buffer for Jacobians."
    result::TR
    "Gradient configuration."
    cfg::TC
end

"""
$(SIGNATURES)

Wrap an ``ℝⁿ`` function `f` in a callable that can be used in [`trust_region_solver`](@ref)
directly. Remaining parameters are passed on to `ForwardDiff.JacobianConfig`, and can be
used to set eg chunk size.

```jldoctest
julia> f(x) = [1.0 2; 3 4] * x - ones(2)
f (generic function with 1 method)

julia> ff = ForwardDiff_wrapper(f, 2)
AD wrapper via ForwardDiff for ℝⁿ→ℝⁿ function, n = 2

julia> ff(ones(2))
(r = [2.0, 6.0], J = [1.0 2.0; 3.0 4.0])

julia> trust_region_solver(ff, [100.0, 100.0])
Nonlinear solver using trust region method converged after 9 steps
  with ‖x‖₂ = 3.97e-15, Δ = 128.0
  x = [-1.0, 1.0]
  r = [-1.78e-15, 3.55e-15]
```
"""
function ForwardDiff_wrapper(f, n, jacobian_config...)
    x = zeros(n)
    result = DiffResults.JacobianResult(x)
    cfg = ForwardDiff.JacobianConfig(f, x, jacobian_config...)
    ForwardDiffBuffer(f, result, cfg)
end

function Base.show(io::IO, ::MIME"text/plain", fdb::ForwardDiffBuffer)
    n = length(DiffResults.value(fdb.result))
    print(io, "AD wrapper via ForwardDiff for ℝⁿ→ℝⁿ function, n = $(n)")
end

function (fdb::ForwardDiffBuffer)(x)
    @unpack f, result, cfg = fdb
    ForwardDiff.jacobian!(result, f, x, cfg)
    (residual = copy(DiffResults.value(result)),
     Jacobian = copy(DiffResults.jacobian(result)))
end

end # module
