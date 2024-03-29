using LinearAlgebra, Test, TrustRegionMethods, UnPack, Logging, NonlinearTestProblems,
    Random, DataFrames
import Optim

# log debug messages
global_logger(SimpleLogger(stdout, Logging.Debug))

# consistent random runs
Random.seed!(1)

include("test_building_blocks.jl")
include("test_problems.jl")

# display diagnostics
@info "iterations" linear_average_iterations nonlinear_iterations
