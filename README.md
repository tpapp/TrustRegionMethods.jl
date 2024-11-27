# TrustRegionMethods.jl

![lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)
[![build](https://github.com/tpapp/TrustRegionMethods.jl/workflows/CI/badge.svg)](https://github.com/tpapp/TrustRegionMethods.jl/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/github/tpapp/TrustRegionMethods.jl/graph/badge.svg?token=Tds39dbcz1)](https://codecov.io/github/tpapp/TrustRegionMethods.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

A simple, and somewhat experimental Julia package for trust region methods, with an emphasis on

1. *Clean functional style*: no preallocated buffers, resulting in less complicated code.

2. *A simple modular interface*: iterate stepwise, or use a simple wrapper.

3. *AD via [DifferentiationInterface](https://juliadiff.org/DifferentiationInterface.jl/DifferentiationInterface/stable/)*:
   harness the power of Julia's AD ecosystem in a simple way.

4. *Support for bailing out*: some inputs just may not be possible or worthwhile to evaluate for very complicated functions (eg economic models). You can signal this by returning non-finite residuals (eg `NaN`s) early.

## Example

```julia
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

## Related packages

This package is very experimental — the interface will be evolving without prior warning or deprecation. You may want to consider the packages below instead.

- [NLsolve.jl](https://github.com/JuliaNLSolvers/NLsolve.jl) is much more mature, but written with a lot of emphasis on using pre-allocated buffers.

- [TRS.jl](https://github.com/oxfordcontrol/TRS.jl) solves trust region subproblems for large scale problems using the generalized eigenvalue solver of Adachi et al (2017). This solver is also implemented in this package, but not optimized for large-scale sparse problems.

## References

See [CITATIONS.bib](CITATIONS.bib).
