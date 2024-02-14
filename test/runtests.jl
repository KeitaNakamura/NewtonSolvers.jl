using NewtonSolvers
using LinearAlgebra
using Test

@testset "NewtonSolvers.solve!" begin
    function f(x)
        f1 = (x[1]+3)*(x[2]^3-7)+18
        f2 = sin(x[2]*exp(x[1])-1)
        [f1, f2]
    end

    function j(x)
        j11 = x[2]^3-7
        j12 = 3*x[2]^2*(x[1]+3)
        u = exp(x[1])*cos(x[2]*exp(x[1])-1)
        j21 = x[2]*u
        j22 = u
        [j11 j12
         j21 j22]
    end

    for T in (Float64, Float32)
        for backtracking in (true, false) # TODO: good tests for backtracking
            x = T[0.1,1.2]
            ch = NewtonSolvers.solve!(f, j, x; backtracking)
            @test ch.isconverged
            @test norm(f(x)) < sqrt(eps(T))
        end
    end

    @testset "Termination" begin
        for T in (Float64, Float32)
            ϵ = sqrt(eps(T))
            z = zero(T)
            for backtracking in (true, false)
                for (sym, f_tol, x_tol, dx_tol) in [(:f,ϵ,z,z), (:x,z,ϵ,z), (:dx,z,z,ϵ)]
                    x = T[0.1,1.2]
                    ch = NewtonSolvers.solve!(f, j, x; f_tol, x_tol, dx_tol, backtracking)
                    @test ch.isconverged
                    @test only(getproperty(ch, sym)) < ϵ
                end
            end
        end
    end

    @testset "Convergence history" begin
        ch = NewtonSolvers.solve!(f, j, [0.1,1.2])
        @test length(ch.f) == 1
        @test length(ch.x) == 1
        @test length(ch.dx) == 1
        ch = NewtonSolvers.solve!(f, j, [0.1,1.2]; logall=true)
        @test length(ch.f) == ch.iters
        @test length(ch.x) == ch.iters
        @test length(ch.dx) == ch.iters
    end

    @testset "Parameters" begin
        @test NewtonSolvers.solve!((x,p)->f(x), (x,p)->j(x), [0.1,1.2], :p).isconverged
        @test_throws MethodError NewtonSolvers.solve!(f, j, [0.1,1.2], :p)
    end

    @testset "Misc" begin
        for T in (Float32, Float64)
            @test (@inferred NewtonSolvers.fx_eltype(f,T[0.1,1.2],NewtonSolvers.NoParameters())) === T
            @test (@inferred NewtonSolvers.fx_eltype((x,p)->f(x),T[0.1,1.2],:p)) === T
        end
    end
end
