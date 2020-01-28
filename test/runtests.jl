using LinearAlgebra, Test, TrustRegionMethods, UnPack
import Optim

include("test_building_blocks.jl")

@testset "linear problem" begin
    for _ in 1:100
        n = rand(2:10)
        J = randn(n, n)
        x0 = randn(n)
        b = J * x0
        res = trust_region_solver(x -> (residual = J * x .- b, Jacobian = J), x0 .* 1000)
        @test res.x ≈ x0 atol = √eps() * n
        @test norm(res.fx.residual, 2) ≈ 0 atol = √eps()
        @test norm(res.fx.residual, 2) == res.residual_norm
        @test res.fx.Jacobian == J
        @test res.converged == true
        display(res)
    end
end

include("test_nonlinear_problems.jl")
