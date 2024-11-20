#####
##### unit tests for building blocks
#####

using TrustRegionMethods: local_residual_model, LocalModel, cauchy_point, dogleg_boundary,
    ges_kernel, solve_model, unconstrained_optimum

"Return a closure that evaluates to the objective function of a model."
function model_objective(model::LocalModel)
    (; f, g, B) = model
    function(p)
        f + dot(p, g) + 0.5 * dot(p, B, p)
    end
end

"Basic sanity check for solver results (splat into tail arguments)."
function is_consistent_solver_results(Δ, p, p_norm, on_boundary)
    all(isfinite, p) &&
        (0 ≤ p_norm ≤ Δ) &&
        (norm(p, 2) ≈ p_norm) &&
        ((p_norm ≈ Δ) == on_boundary)
end

@testset "Nonlinear model constructor sanity checks" begin
    r = ones(2)
    J = ones(2, 2)
    @test_throws ArgumentError local_residual_model(fill(Inf, 2), J)
    @test_throws ArgumentError local_residual_model(fill(NaN, 2), J)
    @test_throws ArgumentError local_residual_model(r, fill(-Inf, 2, 2))
    @test_throws ArgumentError local_residual_model(r, fill(NaN, 2, 2))
    @test_throws DimensionMismatch local_residual_model(r, ones(3, 3))
end

@testset "Cauchy point" begin
    for _ in 1:100
        # random parameters
        n = rand(2:10)
        r = randn(n)
        J = randn(n, n)
        Δ = abs(randn())
        model = local_residual_model(r, J)
        m_obj = model_objective(model)

        # brute force calculation
        pS = - Δ .* normalize(J' * r, 2)
        opt = Optim.optimize(τ -> m_obj(pS .* τ), 0, 1, Optim.Brent())

        # calculate and compare
        pC, _, _= pC_results = @inferred cauchy_point(Δ, model)
        @test is_consistent_solver_results(Δ, pC_results...)
        @test pC ≈ (Optim.minimizer(opt) .* pS) atol = eps()^0.25 * n

        # estimated decrease invariant
        g_norm = norm(J' * r, 2)
        @test m_obj(zeros(n)) - m_obj(pC) ≥ 0.5 * g_norm * min(Δ, g_norm / norm(J' * J, 2))
    end
end

@testset "dogleg quadratic" begin
    for _ in 1:100
        n = rand(2:10)
        pC = randn(n)
        D = randn(n)
        while dot(D, pC) ≤ 0
            D = randn(n)
        end
        τ = rand()
        Δ = norm(pC .+ τ .* D, 2)
        @test @inferred(dogleg_boundary(Δ, D, pC)) ≈ τ
    end
end

@testset "eigensolver type stability" begin
    # we test this separately because it makes it easier to debug type inference failures
    n = 10
    model = local_residual_model(rand(n), rand(n, n))
    λ, gap, y1, y2 = @inferred ges_kernel(1.0, model, I)
    @test λ isa Float64
    @test gap::Float64 ≥ 0
    @test y1 isa Vector{Float64} && y2 isa Vector{Float64}
end

@testset "trust region solver tests" begin
    for _ in 1:100

        # random problem
        n = rand(2:10)
        model = local_residual_model(rand(n), rand(n, n))
        Δ = abs(randn())
        m_obj = model_objective(model)

        # unconstrained optimum and Cauchy point
        pU, pU_norm = @inferred unconstrained_optimum(model)
        @test pU_norm ≥ 0
        pC, _, _ = pC_results = cauchy_point(Δ, model)
        @test is_consistent_solver_results(Δ, pC_results...)

        # Dogleg
        pD, _, _, = pD_results = @inferred solve_model(Dogleg(), Δ, model)
        @test is_consistent_solver_results(Δ, pD_results...)
        @test m_obj(pD) ≤ m_obj(pC) # improve on Cauchy point

        # GES
        pG, _, _ = pG_results = @inferred solve_model(GeneralizedEigenSolver(), Δ, model)
        @test is_consistent_solver_results(Δ, pG_results...)
        @test m_obj(pG) ≤ m_obj(pD) # improve on dogleg
    end
end

@testset "singularities" begin
    # just some basic sanity check to see if the solver can deal with these
    singular1 = local_residual_model(ones(2), ones(2, 2))
    Δ = 1.0
    @test is_consistent_solver_results(Δ, cauchy_point(1.0, singular1)...)
    @test is_consistent_solver_results(Δ, solve_model(Dogleg(), 1.0, singular1)...)
    # FIXME: solver below could do better, just that hard case is not implemented.
    # check for that when it is.
    @test is_consistent_solver_results(Δ, solve_model(GeneralizedEigenSolver(), 1.0,
                                                      singular1)...)
end

@testset "ForwardDiff wrapper" begin
    @testset "different input types" begin
        r = 0:3
        f(x) = x .+ r
        ff = ForwardDiff_wrapper(f, 4)
        J = Diagonal(ones(4))
        for x ∈ (randn(4), zeros(Int, 4), range(1//5, 3//7; length = 4))
            @test ff(x) == (residual = Float64.(x .+ r), Jacobian = J)
        end
    end

    @testset "handle infeasible" begin
        ff = ForwardDiff_wrapper(x -> all(x .> 0) ? x : x .+ NaN, 4)
        @test !all(isfinite, ff(-ones(4)).residual)
        @test ff(ones(4)) == (residual = ones(4), Jacobian = Diagonal(ones(4)))
    end
end

@testset "printing and debug" begin       # just test that printing is defined
    _iterations = -1
    function _debug(args)
        _iterations = args.iterations
    end
    ff = ForwardDiff_wrapper(x -> Diagonal(ones(2)) * x, 2)
    @test repr(ff) isa AbstractString
    res = trust_region_solver(ff, ones(2); debug = _debug)
    @test _iterations == res.iterations
    @test repr(res) isa AbstractString
    @test repr(TrustRegionResult(1, [1.0], nothing, 1.0, false, 99)) isa AbstractString
end
