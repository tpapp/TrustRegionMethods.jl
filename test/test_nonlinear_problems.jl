#####
##### test problems
#####


@testset "solver tests" begin
    for f in TEST_FUNCTIONS
        for local_method in (Dogleg(), GeneralizedEigenSolver())
            @info "solver test" f local_method
            if false            # condition on broken tests here
                @warn "skipping because broken"
                continue
            end
            res = trust_region_solver(ForwardDiff_wrapper(f, dimension(f)), start(f);
                                      local_method = local_method)
            @test res.converged
            @test res.x ≈ root(f) atol = 1e-4 * dimension(f)
        end
    end
end
