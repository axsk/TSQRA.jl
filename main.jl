include("apply_a.jl")
include("tensorops.jl")
include("pentane.jl")

using KrylovKit
using Dates: now

function run(;
    grid=biggrid,
    maxiter=30,
    @time "D" v = interacting_system(grid))

    @time "E" D, E = tensor_sqra(v)
    @time "λ" f = eigenfuns(D, E; maxiter)
    return D, E, f
end

lucaflux = 27.714876666666658

function confirm_lucas_eigenvalues()
    D, E = tensor_sqra(system1(biggrid))
    @show v1 = eigenfuns(D, E)[1] * lucaflux

    D, E = tensor_sqra(system2(biggrid))
    @show v2 = eigenfuns(D, E)[1] * lucaflux

    v1, v2
end

function eigenfuns(D::Array{T}, E::Array{T}; maxiter=100, n=5, tol=1e-6) where {T}
    s = Tuple(size(D))

    # VALUEFIX
    inds = vec(D) .> 0

    x0 = rand(T, sum(inds))
    xt = zeros(T, length(D))

    function Q(x)
        GC.gc()
        xt[inds] .= x
        println(now())
        xx = vec(apply_AE(vec(xt), vec(E), s))
        GC.gc()
        xx[inds]
    end

    @time "solving eigenproblem" KrylovKit.eigsolve(
        Q, x0, n, :LR, verbosity=2, tol=tol, maxiter=maxiter, issymmetric=false)
end
