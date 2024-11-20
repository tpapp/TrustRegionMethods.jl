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
            F = trust_region_problem(x -> J * x .- b, x0 .* 1000)
            result = trust_region_solver(F; local_method = l, maximum_iterations = 50)
            display(result)
            @test result.x ≈ x0 atol = √eps() * n
            @test norm(result.residual, 2) ≈ 0 atol = √eps()
            @test norm(result.residual, 2) == result.residual_norm
            @test result.Jacobian == J
            @test result.converged
            ∑iter += result.iterations
        end
    end
    global linear_average_iterations = round(Int, ∑iter / N) # save for display
end

@testset "almost linear problem restricted to SVector" begin
    fS(x::SVector) = SMatrix{2,2}(1.0, 2.0, 3.0, 4.0) * x .- exp.(x)
    # NOTE: this is inferred because ForwardDiff can calculate the chunk size for SVector
    F = @inferred trust_region_problem(fS, SVector(0.0, 0.0))
    result = @inferred trust_region_solver(F)
    @test result.x isa SVector
    @test result.converged
end

@testset "infeasible region" begin
    @testset "bounded away from solution" begin
        # the solution x = 0 is infeasible, but do we get close?
        function f2(x)
            x[1] ≥ 1 ? x : x .+ NaN
        end
        F2 = trust_region_problem(f2, [3.0])
        result = trust_region_solver(F2)
        @test !result.converged
        @test result.x ≈ [1.0]
    end

    @testset "jump over narrow infeasible region" begin
        history = Bool[]
        function f3(x)
            is_feasible = abs(x[1] - 2) ≥ 0.1 # first step takes us here
            push!(history, is_feasible)
            is_feasible ? x .^ 3  : x .+ NaN
        end
        F3 = trust_region_problem(f3, [3.0])
        result = trust_region_solver(F3; Δ = 5)
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
            F = trust_region_problem(f, starting_point(f))
            res = trust_region_solver(F; local_method = local_method)
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
