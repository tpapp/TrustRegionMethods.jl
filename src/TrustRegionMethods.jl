module TrustRegionMethods

export trust_region_solver, TrustRegionResult, ForwardDiff_wrapper, SolverStoppingCriterion,
    Dogleg, GeneralizedEigenSolver

using ArgCheck: @argcheck
import DiffResults
using DocStringExtensions: FIELDS, FUNCTIONNAME, SIGNATURES, TYPEDEF
import ForwardDiff
using KrylovKit: eigsolve
using LinearAlgebra: dot, I, issuccess, lu, norm, Symmetric, UniformScaling
using UnPack: @unpack

####
#### building blocks
####

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
"""
function cauchy_point(Δ::Real, model::ResidualModel)
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

Calculate the unconstrained optimum of the model, return its norm as the second value.

When the second value is *infinite*, the unconstrained optimum should not be used as this
indicates a singular problem.
"""
function unconstrained_optimum(model::ResidualModel)
    @unpack r, J = model
    LU = lu(J; check = false)
    if issuccess(LU)
        pU = -(LU \ r)
        pU, norm(pU, 2)
    else
        ∞ = convert(eltype(LU), Inf)
        fill(∞, length(r)), ∞
    end
end

####
#### solvers
####

"""
`$(FUNCTIONNAME)(method, Δ, model)`

Optimize the `model` with the given constraint `Δ` using `method`. Returns the following
values:

- the optimum,
- the norm of the optimum,
- a boolean indicating whether the constraint binds.
"""
function solve_model end

"""
The dogleg method.

A simple and efficient heuristic for solving local trust region problems. Does not
necessarily find the optimum, but improves on the Cauchy point.
"""
struct Dogleg end

"""
$(SIGNATURES)

Implementats the branch of the dogleg method where it is already determined that the
unconstrained optimum is lies outside the `Δ` ball. Returns the same kind of results as
[`solve_model`](@ref).
"""
function dogleg_implementation(Δ, model, pU, pU_norm)
    pC, pC_norm, on_boundary = cauchy_point(Δ, model)
    if on_boundary || isinf(pU_norm)
        pC, pC_norm, on_boundary
    else
        D = pU .- pC
        τ = dogleg_boundary(Δ, D, pC, pC_norm)
        pC .+ D .* τ, Δ, true
    end
end

function solve_model(::Dogleg, Δ, model::ResidualModel)
    pU, pU_norm = unconstrained_optimum(model)
    if pU_norm ≤ Δ
        pU, pU_norm, false
    else
        dogleg_implementation(Δ, model, pU, pU_norm)
    end
end

####
#### Generalized eigenvalue solver (Adachi et al 2017).
####
#### NOTE:
####
#### - notation mostly follows the paper
####
#### - implementation has ellipsoidal norm in some parts, for future generalizations,
####   currently unused
####
#### - “hard case” is not implemented yet, falls back to dogleg

"""
Generalized eigenvalue solver (Adachi et al 2017).
"""
struct GeneralizedEigenSolver end

"""
$(SIGNATURES)

Kernel for the generalized eigenvalue solver (methods specialize on the ellipsoidal norm
matrix `B`). Solves the `M̃(λ)` pencil in Adachi et al (2017), eg Algorithm 5.1.

Returns

- `λ`, the largest eigenvalue,

- the gap to the next eigenvalue (see note below)

- `y1` and `y2`, the two parts of the corresponding generalized eigenvalue,

All values are real, methods are type stable.

When theorerical assumptions are violated, `gap` will be non-finite and a debug statement is
emitted. No other values should be used in this case.
"""
function ges_kernel(Δ, model::MinimizationModel, B::UniformScaling)
    @unpack g, A = model
    n = length(g)
    G = ((g * g') ./ abs2(Δ))
    M = [-A G; B -A]
    λs, vs, info = eigsolve(M, 2, :LR)
    @argcheck info.converged ≥ 2 "Eigensolver did not converge."
    T = promote_type(Float64, eltype(M))
    ϵ = √eps(T)
    is_practically_real(z) = imag(z) ≤ n*(ϵ*abs(z) + ϵ)
    λ1, λ2 = λs
    v1 = first(vs)
    if is_practically_real(λ1) && all(is_practically_real, v1)
        λ = real(λ1)::T
        gap = abs(λ - λ2)::T
        y = real.(v1)::Vector{T}
        y1 = y[1:n]
        y2 = y[(n + 1):end]
        λ, gap, y1, y2
    else
        @debug "rightmost eigenvalue not real" M λs vs
        ∞ = T(Inf)
        y∞ = fill(∞, n)
        ∞, ∞, y∞, y∞
    end
end

"""
$(SIGNATURES)

Ellipsoidal norm ``\\| x \\|_B = x'Bx``.
"""
ellipsoidal_norm(x, ::UniformScaling) = norm(x, 2)

function solve_model(::GeneralizedEigenSolver, Δ, model::ResidualModel)
    B = I                       # FIXME: we hardcode Euclidean norm for now
    pU, pU_norm = unconstrained_optimum(model)
    if pU_norm < Δ
        pU, pU_norm, false
    else
        model′ = MinimizationModel(model)
        λ, gap, y1, y2 = ges_kernel(Δ, model′, B)
        τ = √(eps(typeof(λ)) / gap)
        if isfinite(gap) && norm(y1, 2) > τ
            # “easy” case, we can generate a candidate
            @unpack g = model′
            p = (-sign(dot(g, y2)) * Δ / ellipsoidal_norm(y1, B)) .* y1
            p, Δ, true
        else
            # FIXME hard case, this needs to be implemented
            # here we bail and fall back to dogleg
            # NOTE this is also the fallback for eigensolver gone wrong
            @debug "hard case not implemented, falling back to dogleg"
            dogleg_implementation(Δ, model, pU, pU_norm)
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

function reduction_ratio(model::ResidualModel, p, r′)
    @unpack r, J = model
    r2 = sum(abs2, r)
    r′2 = sum(abs2, r′)
    ρ = (r2 - r′2) / (r2 - sum(abs2, r .+ J * p))
    isfinite(ρ) ? ρ : -one(ρ)
end

function trust_region_step(parameters::TrustRegionParameters, local_method, f, Δ, x, fx)
    @unpack η, Δ̄ = parameters
    model = ResidualModel(fx.residual, fx.Jacobian)
    p, p_norm, on_boundary = solve_model(local_method, Δ, model)
    x′ = x .+ p
    fx′ = f(x′)
    ρ = reduction_ratio(model, p, fx′.residual)
    Δ′ =
        if ρ < 0.25
            p_norm / 4
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

struct SolverStoppingCriterion{T <: Real}
    residual_norm_tolerance::T
    function SolverStoppingCriterion(residual_norm_tolerance::T) where {T}
        @argcheck residual_norm_tolerance > 0
        new{T}(residual_norm_tolerance)
    end
end

function SolverStoppingCriterion(; residual_norm_tolerance = √eps())
    SolverStoppingCriterion(residual_norm_tolerance)
end

function check_stopping_criterion(nsc::SolverStoppingCriterion, fx)
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
                             local_method = Dogleg(),
                             stopping_criterion = SolverStoppingCriterion(),
                             maximum_iterations = 500,
                             Δ = 1.0)
    fx = f(x)
    iterations = 1
    while true
        Δ, x, fx = trust_region_step(parameters, local_method, f, Δ, x, fx)
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
struct ForwardDiffBuffer{TX,TF,TR,TC}
    "Buffer for inputs."
    x::TX
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

Non-finite residuals will be treated as infeasible (`nothing`).

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
    ForwardDiffBuffer(x, f, result, cfg)
end

function Base.show(io::IO, ::MIME"text/plain", fdb::ForwardDiffBuffer)
    n = length(DiffResults.value(fdb.result))
    print(io, "AD wrapper via ForwardDiff for ℝⁿ→ℝⁿ function, n = $(n)")
end

function (fdb::ForwardDiffBuffer)(y)
    @unpack x, f, result, cfg = fdb
    # copy to our own buffer to work with types other than Float64
    ForwardDiff.jacobian!(result, f, copy!(x, y), cfg)
    residual = copy(DiffResults.value(result))
    if all(isfinite, residual)
        (residual = residual, Jacobian = copy(DiffResults.jacobian(result)))
    else
        nothing
    end
end

end # module
