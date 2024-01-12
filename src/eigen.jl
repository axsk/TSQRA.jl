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
    verbosity=0) where {T}

    s = Tuple(size(D))

    # VALUEFIX
    inds = vec(D) .> 0

    x0 = isnothing(initialguess) ? rand(T, sum(inds)) : initialguess[inds]

    xt = zeros(T, length(D))
    xx = similar(x0)
    E = vec(E)

    function Q(x)
        xt[inds] .= x
        apply_AE!(xx, xt, E, s)
        xx[inds]
    end

    evs, efs, info = KrylovKit.eigsolve(Q, x0, n, :LR; verbosity, tol, maxiter, issymmetric=false)

    # TODO: these are efs of AE not D^inv AE D
    efs = stack(efs)

    return evs, efs, f
end



function timeeigs(Q::QTensor; n=5,
    maxiter=100,
    tol=1e-6,
    verbosity=0)

    @time eigenfuns(Q.D, Q.E; n, verbosity, tol, maxiter)
    Qs = SqraCore.sqra_grid(getpi(D), beta=getbeta())
    @time eigsolve(Qs, n, :LR; verbosity, tol, maxiter, issymmetric=false)
    return
end