####
#### AD shim for ForwardDiff
####

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

Non-finite residuals will be treated as infeasible.

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
    y = f(x)
    result = DiffResults.JacobianResult(y, x)
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
    (residual = residual, Jacobian = copy(DiffResults.jacobian(result)))
end
