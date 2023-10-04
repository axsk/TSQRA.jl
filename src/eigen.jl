using KrylovKit

""" tensor_sqra(v; clip=Inf)
compute the spectrum of the TSQRA given by the tensors D and E.
Uses a fill-in approach to deal with D.==0 entries and calls the apply_AE routine.
Note that the result only contains the rescaled eigenvectors (apply D.^(-1) to the result).
returns eigenvals, eigenfuns, eigsolve_result
"""
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