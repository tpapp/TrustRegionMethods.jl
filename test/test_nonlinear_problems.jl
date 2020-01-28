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

"Check basic consistency of a problem definition."
function check_problem(f)
    x = root(f)
    @test length(x) == dimension(f)
    @test norm(f(x), 2) ≤ eps()
    x′ = start(f)
    @test length(x′) == length(x)
    @test norm(f(x′), 2) > eps()
end

"Test the solver using problem `f`, using `ForwardDiff` for automatic differentiaton."
function test_solver(f)
    res = trust_region_solver(ForwardDiff_wrapper(f, dimension(f)), start(f))
    @test res.converged
    @test res.x ≈ root(f) atol = √eps() * dimension(f)
end

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

check_problem(F_NWp281())
test_solver(F_NWp281())
