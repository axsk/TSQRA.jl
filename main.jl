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
    function Q(x)
        x = reshape(x, size(E)...)
        println(now())
        x = apply_Q(x, E, D)
        vec(x)
    end

    x = rand(T, prod(size(E)))

    @time "solving eigenproblem" KrylovKit.eigsolve(
        Q,
        x,
        3,
        :LR,
        verbosity=3,
        tol=1e-4,
        maxiter=100)
end

