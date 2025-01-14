@testset "tuning parameters" begin
    @test TrustRegionParameters(0.05, 1) == TrustRegionParameters(0.05, 1.0)
    @test_throws ArgumentError TrustRegionParameters(0.7, 1.0)
    @test_throws ArgumentError TrustRegionParameters(0.1, -1.0)
    @test_throws ArgumentError SolverStoppingCriterion(; residual_norm = -0.1)
    @test_throws ArgumentError SolverStoppingCriterion(; absolute_coordinate_change = -0.1)
    @test_throws ArgumentError SolverStoppingCriterion(; relative_coordinate_change = -0.1)
    @test_throws ArgumentError SolverStoppingCriterion(; absolute_residual_change = -0.1)
    @test_throws ArgumentError SolverStoppingCriterion(; relative_residual_change = -0.1)
end

@testset "printing and debug" begin       # just test that printing is defined
    _iterations = -1
    function _debug(args)
        _iterations = args.iterations
    end
    ff = trust_region_problem(x -> Diagonal(ones(2)) * x, ones(2))
    @test occursin("trust region problem", repr(MIME("text/plain"), ff))
    res = @inferred trust_region_solver(ff; debug = _debug)
    @test res.stop_cause ≠ TrustRegionMethods.StopCause.MaximumIterations
    @test _iterations == res.iterations
    @test occursin("stopped with", repr(MIME("text/plain"), res))
    @test occursin("maximum iterations",
                   repr(MIME("text/plain"),
                        TrustRegionResult(1.0, [1.0], [1.0], [1.0;;], (; residual_norm = 1.0),
                                          TrustRegionMethods.StopCause.MaximumIterations, 99)))
end
