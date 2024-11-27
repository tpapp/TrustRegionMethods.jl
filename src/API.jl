#####
##### high level API
#####

export trust_region_problem, trust_region_solver, TrustRegionParameters, TrustRegionResult,
    SolverStoppingCriterion

####
#### problem definition API
####

"""
$(TYPEDEF)

A container for a struct region problem. Create with [`trust_region_problem`](@ref).

Internal, not part of the API.
"""
struct TrustRegionProblem{TF,TX,TA,TP}
    "The function we are solving for ``f(x) ≈ 0`"
    f::TF
    "The initial `x`, also used to create `AD_prep`"
    initial_x::TX
    "The AD backend, provided by the user"
    AD_backend::TA
    "preparation for AD"
    AD_prep::TP
end

function Base.show(io::IO, ::MIME"text/plain", F::TrustRegionProblem)
    (; f, initial_x, AD_backend, AD_prep) = F
    print(io, "trust region problem",
          "\n  residual dimension: ", length(f(initial_x)),
          "\n  initial x: ", initial_x,
          "\n  AD backend: ", AD_backend)
end

"""
$(SIGNATURES)

Define a trust region problem for solving ``f(x) ≈ 0``, with `initial_x`.

`f` should map vectors to vectors, not necessarily the same size, but the dimension of
the output be as large as that of the input.

`initial_x` should be an `::AbstractVector{T}` type that is closed under addition and
elementwise multiplication by type `T`. For example, if `initial_x::Vector{Float64}` or
`initial_x::SVector{N,Float64}`, then it should be sufficient if `f` handles that, if
not, please open an issue.

!!! NOTE
    For optimal performance, specify the details of the `AD_backend` argument, eg chunk
    size for `ForwardDiff`.
"""
function trust_region_problem(f, initial_x; AD_backend = AutoForwardDiff())
    if !(eltype(initial_x) <: AbstractFloat)
        initial_x = float.(initial_x)
    end
    AD_prep = prepare_jacobian(f, AD_backend, initial_x)
    TrustRegionProblem(f, initial_x, AD_backend, AD_prep)
end

"""
$(TYPEDEF)

Internal representation of the function and the Jacobian evaluated at a particular `x`.
Not part of the API.

# Fields

$(FIELDS)
"""
struct ∂FX{TX,TV,TJ}
    x::TX
    residual::TV
    Jacobian::TJ
end

"""
$(SIGNATURES)

Evaluate the function and the Jacobian at `x`, returning a [`∂FX`](@ref) object.
"""
function evaluate_∂F(F::TrustRegionProblem{TF,TX}, x::T) where {TF,TX,T}
    (; f, AD_backend, AD_prep) = F
    if !(T ≡ TX)
        x = convert(TX, x)::TX
    end
    residual, Jacobian = value_and_jacobian(f, AD_prep, AD_backend, x)
    if all(isfinite, residual)
        @argcheck all(isfinite, Jacobian) "Infinite Jacobian for finite residual."
    end
    ∂FX(x, residual, Jacobian)
end


####
#### trust region steps
####

struct TrustRegionParameters{T}
    η::T
    Δ̄::T
    @doc """
    $(SIGNATURES)

    Trust region method parameters.

    - `η`: trust reduction threshold
    - `Δ̄`: initial trust region radius
    """
    function TrustRegionParameters(η::T, Δ̄::T) where {T <: Real}
        @argcheck 0 < η < 0.25
        @argcheck Δ̄ > 0
        new{T}(η, Δ̄)
    end
end

TrustRegionParameters(η, Δ̄) = TrustRegionParameters(promote(η, Δ̄)...)

TrustRegionParameters(; η = 0.125, Δ̄ = Inf) = TrustRegionParameters(η, Δ̄)

"""
$(SIGNATURES)

Take a trust region step using `local_method`.

`f` is the function that returns the residual and the Jacobian (see
[`trust_region_solver`](@ref)).

`Δ` is the trust region radius, `x` is the position, `∂fx = evaluate_∂F(x)`. Caller
ensures that the latter is feasible.
"""
function trust_region_step(parameters::TrustRegionParameters, local_method,
                           F::TrustRegionProblem, Δ, ∂fx::∂FX)
    (; η, Δ̄) = parameters
    model = local_residual_model(∂fx.residual, ∂fx.Jacobian)
    p, p_norm, on_boundary = solve_model(local_method, Δ, model)
    x′ = ∂fx.x .+ p
    ∂fx′ = evaluate_∂F(F, x′)
    # NOTE: non-finite residuals are handled in reduction_ratio as ρ < 0
    ρ = reduction_ratio(model, p, residual_minimand(∂fx′.residual))
    Δ′ =
        if ρ < 0.25
            p_norm / 4
        elseif ρ > 0.75 && on_boundary
            min(2 * Δ, Δ̄)
        else
            Δ
        end
    if ρ ≥ η
        Δ′, ∂fx′                # reasonable reduction, use new position
    else
        Δ′, ∂fx                 # very small reduction, try again from original
    end
end

struct SolverStoppingCriterion{T <: Real}
    residual_norm_tolerance::T
    @doc """
    $(SIGNATURES)

    Stopping criterion for trust region colver.

    Arguments:

    - `residual_norm_tolerance`: convergence is declared when the norm of the residual is
      below this value
    """
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
    (converged = converged , residual_norm = r_norm)
end

"""
$(TYPEDEF)

A container to return the result of [`trust_region_solver`](@ref). Fields are part of
the API and can be accessed by the user.

# Fields

$(FIELDS)
"""
struct TrustRegionResult{T<:Real,TX<:AbstractVector{T},TR,TJ}
    "The final trust region radius."
    Δ::T
    "The last value (the root only when converged)."
    x::TX
    "`f(x)` at `x`."
    residual::TR
    "`∂f/∂x at `x`."
    Jacobian::TJ
    "The Euclidean norm of the residual at `x`."
    residual_norm::T
    "A boolean indicating convergence."
    converged::Bool
    "Number of iterations (≈ number of function evaluations)."
    iterations::Int
end

function TrustRegionResult(; Δ::T1, x::AbstractVector{T2},
                           residual::AbstractVector{T3},
                           Jacobian::AbstractMatrix{T4},
                           residual_norm::T5, converged,
                           iterations) where {T1 <: Real, T2 <: Real, T3 <: Real,
                                              T4 <: Real, T5 <: Real}
    T = promote_type(T1, T2, T3, T4, T5)
    TrustRegionResult(T(Δ), T.(x), T.(residual), T.(Jacobian), T(residual_norm),
                      converged, iterations)
end

function Base.show(io::IO, ::MIME"text/plain", trr::TrustRegionResult)
    (; Δ, x, residual, Jacobian, residual_norm, converged, iterations) = trr
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
    print(io, "  r = ", _sig.(residual))
end

"""
$(SIGNATURES)

Solve `f ≈ 0` using trust region methods, starting from `x`. Should be provided with a problem
wrapper, see [`trust_region_problem`](@ref).

Returns a [`TrustRegionResult`](@ref) object.

# Keyword arguments (with defaults)

- `parameters = TrustRegionParameters()`: parameters for the trust region method

- `local_method = Dogleg()`: the local method to use

- `stopping_criterion = SolverStoppingCriterion()`: the stopping criterion

- `maximum_iterations = 500`: the maximum number of iterations before declaring
  non-convergence,

- `Δ = 1.0`, the initial trust region radius

- `debug = nothing`: when `≢ nothing`, a function that will be called with an object that
  has properties `iterations, Δ, x, residual, Jacobian, converged, residual_norm`.

# Example

```jldoctest
julia> using TrustRegionMethods

julia> const A = [1.0 2.0; 3.0 4.0]
2×2 Matrix{Float64}:
 1.0  2.0
 3.0  4.0

julia> f(x) = A * x .- exp.(x);

julia> F = trust_region_problem(f, zeros(2))
trust region problem
  residual dimension: 2
  initial x: [0.0, 0.0]
  AD backend: ADTypes.AutoForwardDiff()

julia> result = trust_region_solver(F)
Nonlinear solver using trust region method converged after 5 steps
  with ‖x‖₂ = 1.26e-15, Δ = 1.0
  x = [-0.12, 0.503]
  r = [-8.88e-16, -8.88e-16]

julia> result.converged
true

julia> result.x
2-element Vector{Float64}:
 -0.11979242665753244
  0.5034484917613987
```
"""
function trust_region_solver(F::TrustRegionProblem;
                             parameters = TrustRegionParameters(),
                             local_method = Dogleg(),
                             stopping_criterion = SolverStoppingCriterion(),
                             maximum_iterations = 500,
                             Δ = 1.0,
                             debug = nothing)
    @argcheck Δ > 0
    ∂fx = evaluate_∂F(F, F.initial_x)
    iterations = 1
    while true
        Δ, ∂fx = trust_region_step(parameters, local_method, F, Δ, ∂fx)
        (; x, residual, Jacobian) = ∂fx
        iterations += 1
        (; converged, residual_norm) = check_stopping_criterion(stopping_criterion, ∂fx)
        if debug ≢ nothing
            debug((; iterations, Δ, x, residual, Jacobian, converged, residual_norm))
        end
        reached_max_iter = iterations ≥ maximum_iterations
        if converged || reached_max_iter
            return TrustRegionResult(; Δ, x, residual, Jacobian, residual_norm,
                                     converged, iterations)
        end
    end
end
