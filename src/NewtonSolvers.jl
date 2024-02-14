module NewtonSolvers

using LinearAlgebra

struct ConvergenceHistory{T, U}
    iters::Int
    isconverged::Bool
    f::Vector{T}
    x::Vector{U}
    dx::Vector{U}
end

struct NoParameters
end

function fx_eltype(f,x,p)
    T = Base._return_type(f, Tuple{typeof(x),typeof(p)})
    T == Union{} && f(x,p)
    eltype(T)
end
function fx_eltype(f,x,::NoParameters)
    T = Base._return_type(f, Tuple{typeof(x)})
    T == Union{} && f(x)
    eltype(T)
end

apply(f,x,p) = f(x,p)
apply(f,x,::NoParameters) = f(x)

function solve!(
        f, j, x::AbstractVector, p=NoParameters();
        f_tol::Real=convert(fx_eltype(f,x,p), 1e-8), x_tol::Real=zero(eltype(x)), dx_tol::Real=zero(eltype(x)),
        iterations::Int=1000, linsolve=(x,A,b)->x.=A\b,
        backtracking::Bool=true, showtrace::Bool=false,
        logall::Bool=false,
    )
    compact(val) = rpad(sprint(show, val; context = :compact=>true), 11)
    f!(F, x) = copyto!(F, apply(f,x,p))

    δx = Vector{eltype(x)}(undef, length(x))
    F = Vector{fx_eltype(f,x,p)}(undef, length(x))

    f_hist = eltype(F)[]
    x_hist = eltype(x)[]
    dx_hist = eltype(x)[]

    # compute current residual
    f!(F, x)
    norm(F, Inf) ≤ f_tol && return ConvergenceHistory(0, true, f_hist, x_hist, dx_hist)

    x_prev = copy(x)

    function ϕ(α)
        @. x = x_prev + α * δx
        f!(F, x)
        norm(F)
    end

    @inbounds for i in 1:iterations
        # solve linear system
        linsolve(δx, apply(j,x,p), F)
        rmul!(δx, -1)

        if backtracking == true
            α₀ = one(eltype(δx))
            ϕ_0 = norm(F)
            ϕ′_0 = -ϕ_0
            _backtracking(ϕ, α₀, ϕ_0, ϕ′_0)
        else
            ϕ(1)
        end

        f★ = norm(F, Inf)
        x★ = norm(δx, Inf)
        dx★ = norm(x-x_prev, 2)
        if logall
            push!(f_hist, f★)
            push!(x_hist, x★)
            push!(dx_hist, dx★)
        end

        showtrace && println("|f(x)|∞ = ", compact(f★), "  ",
                             "|δx|∞ = ", compact(x★), "  ",
                             "|x-x'|₂ = ", compact(dx★))

        if f★ ≤ f_tol || x★ ≤ x_tol || dx★ ≤ dx_tol
            if !logall
                push!(f_hist, f★)
                push!(x_hist, x★)
                push!(dx_hist, dx★)
            end
            return ConvergenceHistory(i, true, f_hist, x_hist, dx_hist)
        end

        x_prev .= x
    end

    ConvergenceHistory(iterations, false, f_hist, x_hist, dx_hist)
end

function _backtracking(ϕ, α₀::T, ϕ_0, ϕ′_0) where {T}
    c₁ = T(1e-4)
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
        α₂ = clamp(α★, T(0.1)*α₂, T(0.5)*α₂)

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
