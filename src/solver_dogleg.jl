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

Implements the branch of the dogleg method where it is already determined that the
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

function solve_model(::Dogleg, Δ, model)
    pU, pU_norm = unconstrained_optimizer(model)
    if pU_norm ≤ Δ
        pU, pU_norm, false
    else
        dogleg_implementation(Δ, model, pU, pU_norm)
    end
end
