#####
##### unit tests for building blocks
#####

using TrustRegionMethods: NonlinearModel, cauchy_point, dogleg_boundary, dogleg

"Return a closure that evaluates to the objective function of a model."
function model_objective(model::NonlinearModel)
    @unpack r, J = model
    function(p)
        p' * J' * r + 1/2 * p' * J' * J * p
    end
end

@testset "Nonlinear model constructor sanity checks" begin
    r = ones(2)
    J = ones(2, 2)
    @test_throws ArgumentError NonlinearModel(fill(Inf, 2), J)
    @test_throws ArgumentError NonlinearModel(fill(NaN, 2), J)
    @test_throws ArgumentError NonlinearModel(r, fill(-Inf, 2, 2))
    @test_throws ArgumentError NonlinearModel(r, fill(NaN, 2, 2))
    @test_throws ArgumentError NonlinearModel(r, ones(2, 3))
    @test_throws ArgumentError NonlinearModel(r, ones(3, 3))
end

@testset "Cauchy point" begin
    for _ in 1:100
        # random parameters
        n = rand(2:10)
        r = randn(n)
        J = randn(n, n)
        Δ = abs(randn())
        model = NonlinearModel(r, J)
        m_obj = model_objective(model)

        # brute force calculation
        pS = - Δ .* normalize(model.J' * model.r, 2)
        opt = Optim.optimize(τ -> m_obj(pS .* τ), 0, 1, Optim.Brent())

        # calculate and compare
        pC, pC_norm, on_boundary = @inferred cauchy_point(Δ, model)
        @test pC ≈ (Optim.minimizer(opt) .* pS) atol = eps()^0.25 * n
        @test norm(pC, 2) ≈ pC_norm
        @test (pC_norm ≈ Δ) == on_boundary

        # estimated decrease invariant
        g_norm = norm(J' * r, 2)
        @test -m_obj(pC) ≥ 0.5 * g_norm * min(Δ, g_norm / norm(J' * J, 2))
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

@testset "dogleg" begin
    for _ in 1:100
        n = rand(2:10)
        model = NonlinearModel(rand(n), rand(n, n))
        Δ = abs(randn())
        pC, _, _ = cauchy_point(Δ, model)
        pD, on_boundary = @inferred dogleg(Δ, model)
        @test (norm(pD, 2) ≈ Δ) == on_boundary # report boundary correctly
        m_obj = model_objective(model)
        @test m_obj(pD) ≤ m_obj(pC) # improve on Cauchy point
    end
end

@testset "singularities" begin
    singular_model = NonlinearModel(ones(2), ones(2, 2))
    pC, pC_norm, _ = cauchy_point(1.0, singular_model)
    @test all(isfinite, pC) && isfinite(pC_norm)
    p, _ = dogleg(1.0, singular_model)
    @test all(isfinite, p)
end

@testset "printing" begin       # just test that printing is defined
    ff = ForwardDiff_wrapper(x -> Diagonal(ones(2)) * x, 2)
    @test repr(ff) isa AbstractString
    res = trust_region_solver(ff, ones(2))
    @test repr(res) isa AbstractString
end
