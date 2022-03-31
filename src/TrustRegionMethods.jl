module TrustRegionMethods

using ArgCheck: @argcheck
import DiffResults
using DocStringExtensions: FIELDS, FUNCTIONNAME, SIGNATURES, TYPEDEF
import ForwardDiff
using KrylovKit: eigsolve
using LinearAlgebra: diag, Diagonal, dot, I, issuccess, lu, norm, Symmetric, UniformScaling
using UnPack: @unpack

include("utilities.jl")
include("subproblem.jl")
include("solver_dogleg.jl")
include("solver_generalized_eigen.jl")
include("highlevel.jl")
include("AD_ForwardDiff.jl")

end # module
