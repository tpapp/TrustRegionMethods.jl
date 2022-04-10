using LinearAlgebra, Test, TrustRegionMethods, UnPack, Logging, NonlinearTestProblems,
    Random
import Optim

# log debug messages
global_logger(SimpleLogger(stdout, Logging.Debug))

# consistent random runs
Random.seed!(1)

include("test_building_blocks.jl")
include("test_problems.jl")
