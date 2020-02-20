using LinearAlgebra, Test, TrustRegionMethods, UnPack, Logging
import Optim

# log debug messages
global_logger(SimpleLogger(stdout, Logging.Debug))

include("test_building_blocks.jl")
include("test_simple_problems.jl")
include("test_nonlinear_problems.jl")
