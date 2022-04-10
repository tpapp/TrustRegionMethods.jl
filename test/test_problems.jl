#####
##### simple problems for testing
#####

@testset "linear problem" begin
    N = 100
    ∑iter = 0
    for _ in 1:100
        n = rand(2:10)
        J = randn(n, n)
        x0 = randn(n)
        b = J * x0
        for l in (Dogleg(), )
            @show l
            result = trust_region_solver(x -> (residual = J * x .- b, Jacobian = J),
                                         x0 .* 1000, local_method = l,
                                         maximum_iterations = 50)
            display(result)
            @test result.x ≈ x0 atol = √eps() * n
            @test norm(result.fx.residual, 2) ≈ 0 atol = √eps()
            @test norm(result.fx.residual, 2) == result.residual_norm
            @test result.fx.Jacobian == J
            @test result.converged
            ∑iter += result.iterations
        end
    end
    global linear_average_iterations = round(Int, ∑iter / N) # save for display
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


@testset "solver tests" begin
    TEST_FUNCTIONS = [F_NWp281(),
                      Rosenbrock(),
                      PowellSingular(),
                      PowellBadlyScaled(),
                      HelicalValley(),
                      Beale()]
    LOCAL_METHODS = [Dogleg(), GeneralizedEigenSolver()]
    iterations = zeros(Int, length(TEST_FUNCTIONS), length(LOCAL_METHODS))
    for (i, f) in enumerate(TEST_FUNCTIONS)
        for (j, local_method) in enumerate(LOCAL_METHODS)
            res = trust_region_solver(ForwardDiff_wrapper(f, domain_dimension(f)),
                                      starting_point(f); local_method = local_method)
            @test res.converged
            @test res.x ≈ root(f) atol = 1e-4 * domain_dimension(f)
            iterations[i, j] = res.iterations
        end
    end
    # print number of iterations
    columns = vcat(Vector{Any}([string.(TEST_FUNCTIONS)]), collect.(eachcol(iterations)))
    names = vcat(["local"], string.(LOCAL_METHODS))
    global nonlinear_iterations = DataFrame(columns, names) # save for display
end
