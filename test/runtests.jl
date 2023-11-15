using NewtonSolvers
using LinearAlgebra
using Test

@testset "NewtonSolvers.solve!" begin
    function f!(F, x)
        F[1] = (x[1]+3)*(x[2]^3-7)+18
        F[2] = sin(x[2]*exp(x[1])-1)
        F
    end

    function j!(J, x)
        J[1, 1] = x[2]^3-7
        J[1, 2] = 3*x[2]^2*(x[1]+3)
        u = exp(x[1])*cos(x[2]*exp(x[1])-1)
        J[2, 1] = x[2]*u
        J[2, 2] = u
    end

    function fj!(F, J, x)
        F !== nothing && f!(F, x)
        J !== nothing && j!(J, x)
    end

    for backtracking in (true, false) # TODO: good tests for backtracking
        x = [0.1,1.2]
        converged = NewtonSolvers.solve!(fj!, zeros(2), zeros(2,2), x; backtracking)
        @test converged
        @test norm(f!(zeros(2), x)) < sqrt(eps(Float64))
    end
end
