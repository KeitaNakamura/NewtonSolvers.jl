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

    for T in (Float64, Float32)
        for backtracking in (true, false) # TODO: good tests for backtracking
            x = T[0.1,1.2]
            ch = NewtonSolvers.solve!(f!, j!, zeros(T,2), zeros(T,2,2), x; backtracking)
            @test ch.isconverged
            @test norm(f!(zeros(T,2), x)) < sqrt(eps(T))
        end
    end

    @testset "Termination" begin
        for T in (Float64, Float32)
            ϵ = sqrt(eps(T))
            for backtracking in (true, false)
                for (f_tol, x_tol, dx_tol) in ([ϵ,0,0], [0,ϵ,0], [0,0,ϵ])
                    x = T[0.1,1.2]
                    ch = NewtonSolvers.solve!(f!, j!, zeros(T,2), zeros(T,2,2), x; f_tol, x_tol, dx_tol, backtracking)
                    @test ch.isconverged
                    @test norm(f!(zeros(T,2), x)) < sqrt(eps(T))
                end
            end
        end
    end

    @testset "Convergence history" begin
        ch = NewtonSolvers.solve!(f!, j!, zeros(2), zeros(2,2), [0.1,1.2])
        @test length(ch.f) == 1
        @test length(ch.x) == 1
        @test length(ch.dx) == 1
        ch = NewtonSolvers.solve!(f!, j!, zeros(2), zeros(2,2), [0.1,1.2]; logall=true)
        @test length(ch.f) == ch.iters
        @test length(ch.x) == ch.iters
        @test length(ch.dx) == ch.iters
    end
end
