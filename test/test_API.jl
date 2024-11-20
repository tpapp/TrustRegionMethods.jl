@testset "tuning parameters" begin
    @test TrustRegionParameters(0.05, 1) == TrustRegionParameters(0.05, 1.0)
    @test_throws ArgumentError TrustRegionParameters(0.7, 1.0)
    @test_throws ArgumentError TrustRegionParameters(0.1, -1.0)

    @test_throws ArgumentError SolverStoppingCriterion(-0.1)
end

@testset "printing and debug" begin       # just test that printing is defined
    _iterations = -1
    function _debug(args)
        _iterations = args.iterations
    end
    ff = trust_region_problem(x -> Diagonal(ones(2)) * x, ones(2))
    @test occursin("trust region problem", repr(MIME("text/plain"), ff))
    res = @inferred trust_region_solver(ff; debug = _debug)
    @test res.converged
    @test _iterations == res.iterations
    @test occursin("converged", repr(MIME("text/plain"), res))
    @test occursin("didn't converge",
                   repr(MIME("text/plain"),
                        TrustRegionResult(1.0, [1.0], [1.0], [1.0;;], 1.0, false, 99)))
end
