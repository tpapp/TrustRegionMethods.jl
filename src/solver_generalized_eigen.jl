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

export GeneralizedEigenSolver

"""
Generalized eigenvalue solver (Adachi et al 2017).
"""
struct GeneralizedEigenSolver end

"""
$(SIGNATURES)

Kernel for the generalized eigenvalue solver (methods specialize on the ellipsoidal norm
matrix `S`). Solves the `M̃(λ)` pencil in Adachi et al (2017), eg Algorithm 5.1.

Returns

- `λ`, the largest eigenvalue,

- the gap to the next eigenvalue (see note below)

- `y1` and `y2`, the two parts of the corresponding generalized eigenvalue,

All values are real, methods are type stable.

When theorerical assumptions are violated, `gap` will be non-finite and a debug statement is
emitted. No other values should be used in this case.
"""
function ges_kernel(Δ, model::LocalModel, S::UniformScaling)
    (; g, B) = model
    n = length(g)
    G = ((g * g') ./ abs2(Δ))
    M = [-B G; S -B]
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

function solve_model(::GeneralizedEigenSolver, Δ, model::LocalModel)
    S = I                       # FIXME: we hardcode Euclidean norm for now
    pU, pU_norm = unconstrained_optimum(model)
    if pU_norm < Δ
        pU, pU_norm, false
    else
        λ, gap, y1, y2 = ges_kernel(Δ, model, S)
        τ = √(eps(typeof(λ)) / gap)
        if isfinite(gap) && norm(y1, 2) > τ
            # “easy” case, we can generate a candidate
            (; g) = model
            p = (-sign(dot(g, y2)) * Δ / ellipsoidal_norm(y1, S)) .* y1
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
