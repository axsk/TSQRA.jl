include("apply_a.jl")
include("tensorops.jl")
include("pentane.jl")

using KrylovKit
using Dates: now

function run()
    @time "precomputing" D, E = sqra_pentane()

    f = eigenfuns(D, E)
    return D, E, f
end

function eigenfuns(D::Array{T}, E::Array{T}) where {T}
    s = Tuple(size(E))

    di = Di(vec(D))
    x = rand(T, length(D))

    function Q(x)
        println(now())
        vec(apply_Q(vec(x), vec(E), vec(D), s, Di=di))
    end

    @time "solving eigenproblem" KrylovKit.eigsolve(
        Q,
        x,
        3,
        :LR,
        verbosity=2,
        tol=1e-4,
        maxiter=10000)
end

