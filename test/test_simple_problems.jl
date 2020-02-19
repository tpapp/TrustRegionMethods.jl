#####
##### simple problems for testing
#####

@testset "linear problem" begin
    for _ in 1:100
        n = rand(2:10)
        J = randn(n, n)
        x0 = randn(n)
        b = J * x0
        for l in (Dogleg(), )
            @show l
            result = trust_region_solver(x -> (residual = J * x .- b, Jacobian = J),
                                         x0 .* 1000, local_method = l)
            @test result.x ≈ x0 atol = √eps() * n
            @test norm(result.fx.residual, 2) ≈ 0 atol = √eps()
            @test norm(result.fx.residual, 2) == result.residual_norm
            @test result.fx.Jacobian == J
            @test result.converged
            display(result)
        end
    end
end

@testset "infeasible region" begin
    # the solution x = 0 is infeasible, but do we get close?
    function f(x)
        (residual = x[1] ≥ 1 ? x : x .+ NaN, Jacobian = Diagonal(ones(length(x))))
    end
    result = trust_region_solver(f, [3.0])
    @test !result.converged
    @test result.x ≈ [1.0]
end
