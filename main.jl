include("apply_a.jl")
include("tensorops.jl")
include("pentane.jl")
include("sens_exp.jl")

using KrylovKit
using Dates: now

function run(;
    grid=biggrid,
    maxiter=30,
    v=@time "D" interacting_system(grid))

    @time "E" D, E = tensor_sqra(v)
    @time "λ" f = eigenfuns(D, E; maxiter)
    return D, E, f
end


lucaflux = 27.714876666666658


function eigenfuns(D::Array{T}, E::Array{T};
    initialguess=nothing,
    n=5,
    maxiter=100,
    tol=1e-6,
    verbosity=1) where {T}

    s = Tuple(size(D))

    # VALUEFIX
    inds = vec(D) .> 0

    x0 = isnothing(initialguess) ? rand(T, sum(inds)) : initialguess[inds]
    xt = zeros(T, length(D))

    function Q(x)
        #GC.gc()
        xt[inds] .= x
        #println(now())
        xx = vec(apply_AE(vec(xt), vec(E), s))
        #GC.gc()
        xx[inds]
    end

    GC.gc()

    f = @time "solving eigenproblem" KrylovKit.eigsolve(
        Q, x0, n, :LR; verbosity, tol, maxiter, issymmetric=false)

    efs = zeros(T, length(D), n)
    for i in 1:n
        efs[inds, i] = f[2][i]
    end

    evs = f[1]
    return evs, efs, f
end

""" construct an initial guess for the interacting systems krylovspace
by computing the eigenfunction of the combined system """
function initialguess(grid=defaultgrid)
    efs = map([system1(grid), system2(grid)]) do sys
        d, e = tensor_sqra(sys)
        vals, efs, = eigenfuns(d, e, n=2)
        real.(efs[:, 2])
    end
    kron(efs...)
end

function visualize(x, g; kwargs...)
    c = [
        g[x[1]], g[x[2]], g[x[3]],  # 1 2 3
        g[x[4]], g[x[5]], 0,  # 4 5 x
        0, 0, 0,  # x x x
        0, g[x[6]], 0,  # x 6 x
        g[x[7]], g[x[8]], g[x[9]]]  # 7 8 9
    c = reshape(c, 3, :)
    plot!(eachrow(c)...; kwargs...)
    scatter!(eachrow(c)...; kwargs...)
end
