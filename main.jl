includet("apply_a.jl")
includet("tensorops.jl")
includet("pentane.jl")

using KrylovKit
using Dates: now

function run(; grid=defaultgrid, iter=30)
    @time "precomputing" D, E = sqra_pentane1(grid)

    f = eigenfuns(D, E; iter)
    return D, E, f
end
lucaflux = 27.714876666666658
function sqra_pentane_new(
    v=vtensor_system1(defaultgrid))
    D = tensor_sqra(v; beta)
    #@assert isapprox(D[1], 0.57709024, rtol=1e-5)
    N = length(D)

    E = apply_Qo(ones(N), vec(D), size(D))

    return D, E
end

function confirm_luca()
    D, E = sqra_pentane_new()
    @show v1 = eigenfuns(D, E)[1] * lucaflux

    D, E = sqra_pentane_new(vtensor_system2())
    @show v2 = eigenfuns(D, E)[1] * lucaflux

    v1, v2
end

function sparse_Q(D, E, maxcol=10)
    function Q(x)
        apply_Q(x, vec(E), vec(D), size(D))
    end
    reconstruct_matrix_sparse(Q, length(D); maxcol)
end

function eigenfuns(D::Array{T}, E::Array{T}; iter=100) where {T}

    s = Tuple(size(D))

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

using SparseArrays
function reconstruct_matrix_sparse(action, len; maxcol=len)
    A = spzeros(len, len)
    for i in 1:maxcol
        x = zeros(len)
        x[i] = 1
        A[:, i] = action(x)
    end
    return A
end