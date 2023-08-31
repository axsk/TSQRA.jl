include("apply_a.jl")
include("tensorops.jl")
include("pentane.jl")

using KrylovKit
using Dates: now

function run(; ngrid=8, iter=30)
    @time "precomputing" D, E = sqra_pentane(; ngrid)

    f = eigenfuns(D, E; iter)
    return D, E, f
end

function eigenfuns(D::Array{T}, E::Array{T}; iter=100) where {T}

    s = Tuple(size(E))

    di = Di(vec(D))

    # VALUEFIX
    inds = vec(D) .> 0

    x = rand(T, sum(inds))
    xt = zeros(T, length(D))



    function Q(x)
        xt[inds] .= x
        #vec(apply_Q(vec(x), vec(E), vec(D), s, Di=di))
        xx = vec(apply_AE(vec(xt), vec(E), s))
        xx[inds]
    end
    print(now())
    @time "solving eigenproblem" KrylovKit.eigsolve(
        Q,
        x,
        4,
        :LR,
        verbosity=2,
        tol=1e-4,
        maxiter=iter,
        issymmetric=true)
end

function eigenfuns(E::Array{T}; iter=100, cutoff=Inf, tol=1e-6, inds=vec(E) .< cutoff, n=4) where {T}
    s = Tuple(size(E))


    println("cutting off $(1-(sum(inds)/length(E))) of all values")

    x = rand(T, sum(inds))
    xt = zeros(T, length(E))

    function Q(x)
        xt[inds] .= x
        #vec(apply_Q(vec(x), vec(E), vec(D), s, Di=di))
        vec(apply_AE(vec(xt), vec(E), s))[inds]
    end

    @time "solving eigenproblem" KrylovKit.eigsolve(
        Q,
        x,
        n,
        :LR,
        verbosity=2,
        tol=tol,
        maxiter=iter,
        issymmetric=true)
end

function EfromD(D)
    Qo(ones(size(D)), D, Di=D)
end
