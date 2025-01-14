#####
##### high level API
#####

export trust_region_problem, trust_region_solver, TrustRegionParameters, TrustRegionResult,
    SolverStoppingCriterion

public StopCause

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

"""
$(SIGNATURES)

The objective of the problem formulated as minimization.
"""
function calculate_objective_reduction(a::∂FX, b::∂FX)
    # NOTE: equivalent to elementwise a^2 - b^2, implemented for numerical stability
    mapreduce((a, b) -> (a + b) * (a - b), +, a.residual, b.residual)
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
    model_reduction = calculate_model_reduction(model, p)
    objective_reduction = calculate_objective_reduction(∂fx, ∂fx′)
    ρ = objective_reduction / model_reduction
    if !isfinite(ρ)
        ρ = -one(ρ)             # handle non-finite residuals
    end
    Δ′ =
        if ρ < 0.25
            p_norm / 4
        elseif ρ > 0.75 && on_boundary
            min(2 * Δ, Δ̄)
        else
            Δ
        end
    take_step = ρ ≥ η           # use new position
    (; Δ = Δ′, ∂fx′ = take_step ? ∂fx′ : ∂fx,
     on_boundary, step = p, step_norm = p_norm, step_taken = take_step,
     objective_reduction, model_reduction)
end

"""
$(SIGNATURES)

Diagnostics for a trust region step.
"""
function trust_region_step_diagnostics(∂fx::∂FX, ∂fx′::∂FX)
    (absolute_residual_change = mapreduce(absolute_difference, max,
                                            ∂fx.residual, ∂fx′.residual),
     relative_residual_change = mapreduce(relative_difference, max,
                                             ∂fx.residual, ∂fx′.residual),
     absolute_coordinate_change = mapreduce(absolute_difference, max,
                                            ∂fx.x, ∂fx′.x),
     relative_coordinate_change = mapreduce(relative_difference, max,
                                            ∂fx.x, ∂fx′.x),
     residual_norm = norm(∂fx′.residual, 2))
end

struct SolverStoppingCriterion{T <: Real}
    residual_norm::T
    absolute_coordinate_change::T
    relative_coordinate_change::T
    absolute_residual_change::T
    relative_residual_change::T
    @doc """
    $(SIGNATURES)

    Stopping criterion for trust region colver. Fields are compared to the values
    obtained from the latest step, and if **any** of the latter is smaller, the solver
    stops.

    Norms are Euclidean, changes in vectors are the maximum of elementwise absolute or
    relative difference.
    """
    function SolverStoppingCriterion(; residual_norm::Real = √eps(),
                                     absolute_coordinate_change::Real = √eps(),
                                     relative_coordinate_change::Real = √eps(),
                                     absolute_residual_change::Real = √eps(),
                                     relative_residual_change::Real = √eps())
        (residual_norm,
         absolute_coordinate_change,
         relative_coordinate_change,
         absolute_residual_change,
         relative_residual_change) = promote(residual_norm,
                                             absolute_coordinate_change,
                                             relative_coordinate_change,
                                             absolute_residual_change,
                                             relative_residual_change)
        @argcheck residual_norm ≥ 0
        @argcheck absolute_coordinate_change ≥ 0
        @argcheck relative_coordinate_change ≥ 0
        @argcheck absolute_residual_change ≥ 0
        @argcheck relative_residual_change ≥ 0
        new{typeof(residual_norm)}(residual_norm,
               absolute_coordinate_change, relative_coordinate_change,
               absolute_residual_change, relative_residual_change)
    end
end

"""
Reason for stopping the solver. See the docstrings of values for each.
"""
@enumx StopCause begin
    "residual norm below the specified tolerance"
    ResidualNorm
    "largest absolute residual change below specified tolerance"
    AbsoluteResidualChange
    "largest relative residual change below specified tolerance"
    RelativeResidualChange
    "largest absolute coordinate change below specified tolerance"
    AbsoluteCoordinateChange
    "largest relative coordinate change below specified tolerance"
    RelativeCoordinateChange
    "reached maximum iterations"
    MaximumIterations
end

"""
$(SIGNATURES)

Check whether we need to stop, and either return an applicable [`StopCause`](@ref), or
`nothing` if there is no reason to stop.
"""
function check_stopping_criterion(nsc::SolverStoppingCriterion, diagnostics)
    if diagnostics.residual_norm ≤ nsc.residual_norm
        return StopCause.ResidualNorm
    end
    if diagnostics.absolute_residual_change ≤ nsc.absolute_residual_change
        return StopCause.AbsoluteResidualChange
    end
    if diagnostics.relative_residual_change ≤ nsc.relative_residual_change
        return StopCause.RelativeResidualChange
    end
    if diagnostics.absolute_coordinate_change ≤ nsc.absolute_coordinate_change
        return StopCause.AbsoluteCoordinateChange
    end
    if diagnostics.relative_coordinate_change ≤ nsc.relative_coordinate_change
        return StopCause.RelativeCoordinateChange
    end
    nothing
end

"""
$(TYPEDEF)

A container to return the result of [`trust_region_solver`](@ref). Fields are part of
the API and can be accessed by the user.

# Fields

$(FIELDS)
"""
struct TrustRegionResult{T<:Real,TX<:AbstractVector{T},TR,TJ,TD}
    "The final trust region radius."
    Δ::T
    "The last value (the root only when converged)."
    x::TX
    "`f(x)` at `x`."
    residual::TR
    "`∂f/∂x at `x`."
    Jacobian::TJ
    "Diagnostics for the last step."
    last_step_diagnostics::TD
    "Reason for stopping."
    stop_cause::StopCause.T
    "Number of iterations (≈ number of function evaluations)."
    iterations::Int
end

function TrustRegionResult(; Δ::T1, x::AbstractVector{T2},
                           residual::AbstractVector{T3},
                           Jacobian::AbstractMatrix{T4},
                           last_step_diagnostics, stop_cause,
                           iterations) where {T1 <: Real, T2 <: Real, T3 <: Real, T4 <: Real}
    T = promote_type(T1, T2, T3, T4)
    TrustRegionResult(T(Δ), T.(x), T.(residual), T.(Jacobian), last_step_diagnostics,
                      stop_cause, iterations)
end

function Base.getproperty(trr::TrustRegionResult, key::Symbol)
    if key ≡ :converged
        trr.stop_cause ≠ StopCause.MaximumIterations
    else
        getfield(trr, key)
    end
end

function Base.show(io::IO, ::MIME"text/plain", trr::TrustRegionResult)
    (; Δ, x, residual, Jacobian, last_step_diagnostics, stop_cause, iterations) = trr
    _sig(x) = round(x; sigdigits = 3)
    print(io, "Nonlinear solver using trust region method ")
    if stop_cause == StopCause.MaximumIterations
        printstyled(io, "reached maximum iterations"; color = :red)
    else
        printstyled(io, "stopped with ", stop_cause; color = :green)
    end
    println(io, " after $(iterations) steps")
    print(io, "  with ")
    (; residual_norm) = last_step_diagnostics
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
    iterations = 0
    while true
        iterations += 1
        (; Δ, ∂fx′, on_boundary, step, step_taken, step,
         step_norm) = trust_region_step(parameters, local_method, F, Δ, ∂fx)
        step_diagnostics = trust_region_step_diagnostics(∂fx, ∂fx′)
        if step_taken
            ∂fx = ∂fx′
            stop_cause = check_stopping_criterion(stopping_criterion, step_diagnostics)
        else
            stop_cause = nothing
        end
        (; x, residual, Jacobian) = ∂fx
        if debug ≢ nothing
            debug(merge((; iterations, Δ, x, residual, Jacobian,
                         step, on_boundary, step_taken, step_norm),
                        step_diagnostics))
        end
        if iterations ≥ maximum_iterations
            stop_cause = something(stop_cause, StopCause.MaximumIterations)
        end
        if stop_cause ≢ nothing
            return TrustRegionResult(; Δ, x, residual, Jacobian,
                                     last_step_diagnostics = step_diagnostics,
                                     stop_cause, iterations)
        end
    end
end
