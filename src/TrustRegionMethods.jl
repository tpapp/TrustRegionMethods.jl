"""
$(DocStringExtensions.README)
"""
module TrustRegionMethods

using ArgCheck: @argcheck
using DifferentiationInterface: prepare_jacobian, value_and_jacobian, AutoForwardDiff
using DocStringExtensions: FIELDS, FUNCTIONNAME, SIGNATURES, TYPEDEF, DocStringExtensions
using EnumX: @enumx
import ForwardDiff
using KrylovKit: eigsolve
using LinearAlgebra: diag, Diagonal, dot, I, issuccess, lu, norm, Symmetric, UniformScaling
using SymmetricProducts: SELF

include("utilities.jl")
include("subproblem.jl")
include("solver_dogleg.jl")
include("solver_generalized_eigen.jl")
include("API.jl")

end # module
