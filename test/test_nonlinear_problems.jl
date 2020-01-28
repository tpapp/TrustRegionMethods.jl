#####
##### test problems
#####

####
#### generic API
####
#### A *test problem* is a `ℝⁿ → ℝⁿ` mapping with some metadata (roots, starting point,
#### dimension).
####

"Dimension of a test problem."
function dimension end

"Root (solution) of a test problem."
function root end

"Recommended starting point of a test problem."
function start end

####
#### specific test problems
####

###
### Nocedal and Wright p 281
###

"Problem from Nocedal and Wright, p 281."
struct F_NWp281 end

dimension(::F_NWp281) = 2

root(::F_NWp281) = [0, 1]

start(::F_NWp281) = [-0.5, 1.4]

function (::F_NWp281)(x)
    x1, x2 = x
    [(x1 + 3) * (x2^3 - 7) + 18, sin(x2 * exp(x1) - 1)]
end

###
### Rosenbrock
###

"Rosenbrock function."
struct Rosenbrock end

dimension(::Rosenbrock) = 2

root(::Rosenbrock) = [1, 1]

start(::Rosenbrock) = [-1.2, 1]

function (::Rosenbrock)(x)
    x1, x2 = x
    [10 * (x2 - abs2(x1)), 1 - x1]
end

###
### Powell singular function
###

struct PowellSingular end

dimension(::PowellSingular) = 4

root(::PowellSingular) = zeros(4)

start(::PowellSingular) = Float64[3, -1, 0, 1]

function (::PowellSingular)(x)
    x1, x2, x3, x4 = x
    [x1 + 10 * x2, √5 * (x3 - x4), abs2(x2 - 2 * x3), √10 * abs2(x1 - x4)]
end

TEST_FUNCTIONS = [F_NWp281(), Rosenbrock(), PowellSingular()]

@testset "basic consistency checks for test functions." begin
    for f in TEST_FUNCTIONS
        x = root(f)
        @test length(x) == dimension(f)
        @test norm(f(x), 2) ≤ eps()
        x′ = start(f)
        @test length(x′) == length(x)
        @test norm(f(x′), 2) > eps()
    end
end

@testset "solver tests" begin
    for f in TEST_FUNCTIONS[3:3]
        res = trust_region_solver(ForwardDiff_wrapper(f, dimension(f)), start(f))
        @test res.converged
        @test res.x ≈ root(f) atol = 1e-4 * dimension(f)
    end
end
