#####
##### Dogleg implementation
#####

export Dogleg

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
