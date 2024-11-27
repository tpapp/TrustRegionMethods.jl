using LinearAlgebra, Test, TrustRegionMethods, Logging, NonlinearTestProblems, Random,
    DataFrames, StaticArrays
import Optim

# log debug messages
global_logger(SimpleLogger(stdout, Logging.Debug))

# consistent random runs
Random.seed!(1)

include("test_building_blocks.jl")
include("test_API.jl")
include("test_problems.jl")

# display diagnostics
@info "iterations" linear_average_iterations nonlinear_iterations

using JET
@testset "static analysis with JET.jl" begin
    @test isempty(JET.get_reports(report_package(TrustRegionMethods,
                                                 target_modules=(TrustRegionMethods,))))
end

@testset "QA with Aqua" begin
    import Aqua
    Aqua.test_all(TrustRegionMethods; ambiguities = false)
    # testing separately, cf https://github.com/JuliaTesting/Aqua.jl/issues/77
    Aqua.test_ambiguities(TrustRegionMethods)
end
