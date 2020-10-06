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
    @testset "bounded away from solution" begin
        # the solution x = 0 is infeasible, but do we get close?
        function f(x)
            (residual = x[1] ≥ 1 ? x : x .+ NaN, Jacobian = Diagonal(ones(length(x))))
        end
        result = trust_region_solver(f, [3.0])
        @test !result.converged
        @test result.x ≈ [1.0]
    end

    @testset "jump over narrow infeasible region" begin
        history = Bool[]
        function f3(x)
            is_feasible = abs(x[1] - 2) ≥ 0.1 # first step takes us here
            push!(history, is_feasible)
            (residual = is_feasible ? x .^ 3  : x .+ NaN, Jacobian = Diagonal(@. 3 * abs2(x)))
        end
        result = trust_region_solver(f3, [3.0]; Δ = 5)
        @test result.converged
        @test result.x ≈ [0.0] atol = 0.03
        @test any(!, history)   # check that infeasible region was visited
    end
end

#####
##### nonlinear test problems
#####

TEST_FUNCTIONS = [F_NWp281(),
                  Rosenbrock(),
                  PowellSingular(),
                  PowellBadlyScaled(),
                  HelicalValley()]

@testset "solver tests" begin
    for f in TEST_FUNCTIONS
        for local_method in (Dogleg(), GeneralizedEigenSolver())
            @info "solver test" f local_method
            if false            # condition on broken tests here
                @warn "skipping because broken"
                continue
            end
            res = trust_region_solver(ForwardDiff_wrapper(f, domain_dimension(f)),
                                      starting_point(f); local_method = local_method)
            @test res.converged
            @test res.x ≈ root(f) atol = 1e-4 * domain_dimension(f)
        end
    end
end
