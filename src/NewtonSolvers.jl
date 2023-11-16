module NewtonSolvers

using LinearAlgebra

function solve!(
        FJ!, F::AbstractVector{T}, J, x::AbstractVector, δx::AbstractVector=fill!(similar(x), zero(eltype(x)));
        atol::T=sqrt(eps(T)), rtol::T=zero(T), maxiter::Int=20, linsolve=(x,A,b)->x.=A\b,
        backtracking::Bool=true, showtrace::Bool=false,
    ) where {T}

    # compute current residual
    FJ!(F, nothing, x)
    f = norm(F)
    TOL = max(rtol*f, atol)
    f < TOL && return true

    x_prev = copy(x)

    function ϕ(α)
        @. x = x_prev + α * δx
        FJ!(F, nothing, x)
        norm(F)
    end

    @inbounds for _ in 1:maxiter
        # solve linear system
        FJ!(nothing, J, x)
        linsolve(δx, J, F)
        rmul!(δx, -1)

        if backtracking == true
            α₀ = one(eltype(δx))
            ϕ_0 = norm(F)
            ϕ′_0 = -ϕ_0
            f = _backtracking(ϕ, α₀, ϕ_0, ϕ′_0)
        else
            f = ϕ(1)
        end

        showtrace && println("| f(x) | = ", f)
        f < TOL && return true

        x_prev .= x
    end

    false
end

function _backtracking(ϕ, α₀, ϕ_0, ϕ′_0)
    c₁ = 1e-4
    α₁ = α₂ = α₀
    ϕ_α₁ = ϕ_α₂ = ϕ(α₀)
    k = 0
    while ϕ_α₂ > ϕ_0 + c₁*α₂*ϕ′_0
        k += 1
        if k > 1000
            @warn "Backtracking not converged, proceed anyway"
            break
        end

        if k == 1
            α★ = backtracking_quadratic_interpolation(α₁, ϕ_0, ϕ′_0, ϕ_α₁)
        else
            α★ = backtracking_cubic_interpolation(α₁, α₂, ϕ_0, ϕ′_0, ϕ_α₁, ϕ_α₂)
        end

        α₁ = α₂
        α₂ = clamp(α★, 0.1α₂, 0.5α₂)

        ϕ_α₁ = ϕ_α₂
        ϕ_α₂ = ϕ(α₂)
    end
    ϕ_α₂
end

function backtracking_quadratic_interpolation(α₀, ϕ_0, ϕ′_0, ϕ_α₀)
    -(ϕ′_0 * α₀^2) / 2(ϕ_α₀ - ϕ_0 - ϕ′_0 * α₀)
end

function backtracking_cubic_interpolation(α₀, α₁, ϕ_0, ϕ′_0, ϕ_α₀, ϕ_α₁)
    x = ϕ_α₁ - ϕ_0 - ϕ′_0*α₁
    y = ϕ_α₀ - ϕ_0 - ϕ′_0*α₀
    γ = α₀^2 * α₁^2 * (α₁-α₀)
    a = (α₀^2*x - α₁^2*y) / γ
    b = (-α₀^3*x + α₁^3*y) / γ
    (-b + √(b^2 - 3*a*ϕ′_0)) / 3a
end

end # module NewtonSolvers
