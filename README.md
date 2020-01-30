# TrustRegionMethods.jl

![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)<!--
![Lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-stable-green.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-retired-orange.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-archived-red.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-dormant-blue.svg) -->
[![Build Status](https://travis-ci.com/tpapp/TrustRegionMethods.jl.svg?branch=master)](https://travis-ci.com/tpapp/TrustRegionMethods.jl)
[![codecov.io](http://codecov.io/github/tpapp/TrustRegionMethods.jl/coverage.svg?branch=master)](http://codecov.io/github/tpapp/TrustRegionMethods.jl?branch=master)

Experimental Julia package for trust region methods, with an emphasis on

1. *Clean functional style*: no preallocated buffers, resulting in less complicated code.

2. *A simple modular interface*: iterate stepwise, or use a simple wrapper.

3. *AD agnostic function evaluations*: an objective function just returns a value with properties `residual` and `Jacobian`. It can be any type that supports this, and carry extra payload relevant to your problem. However, if you just want to code an ℝⁿ → ℝⁿ function, it can do AD for you using wrappers (currently `ForwardDiff`).

4. *Support for bailing out*: some inputs just may not be possible or worthwhile to evaluated for very complicated functions (eg economic models). You can signal this by returning `nothing`.

## Example

```julia
julia> using TrustRegionMethods

julia> f(x) = [1.0 2; 3 4] * x - ones(2) # very simple linear function
f (generic function with 1 method)

julia> ff = ForwardDiff_wrapper(f, 2)    # AD via a wrapper results in a callable
AD wrapper via ForwardDiff for ℝⁿ→ℝⁿ function, n = 2

julia> ff(ones(2))                       # this is what the solver will need
(residual = [2.0, 6.0], Jacobian = [1.0 2.0; 3.0 4.0])

julia> trust_region_solver(ff, [100.0, 100.0]) # remote starting point
Nonlinear solver using trust region method converged after 9 steps
  with ‖x‖₂ = 3.97e-15, Δ = 128.0
  x = [-1.0, 1.0]
  r = [-1.78e-15, 3.55e-15]
```

## Related packages

This package is very experimental — the interface will be evolving without prior warning or deprecation. You may want to consider the packages below instead.

- [NLsolve.jl](https://github.com/JuliaNLSolvers/NLsolve.jl) is much more mature, but written with a lot of emphasis on using pre-allocated buffers.

- [TRS.jl](https://github.com/oxfordcontrol/TRS.jl) solves trust region subproblems for large scale problems using the generalized eigenvalue solver of Adachi et al (2017). This solver is also implemented in this package, but not optimized for large-scale sparse problems.

## References

See <CITATIONS.bib>.
