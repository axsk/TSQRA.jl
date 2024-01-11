using LinearAlgebra: CartesianIndex, pinv
using Random: randexp

using SqraCore
using PCCAPlus
include("tsqra_simple.jl")

Tensor = AbstractArray

""" Gillespie for SQRA via stationary tensor D """
function gillespie(i::CartesianIndex, D::Tensor, T::Float64)
    ni = neighboroffsets(D)
    t = 0.0
    while true
        out = neighborrates(i, D, ni)
        cs = cumsum(out)
        rate = cs[end]
        t += randexp() / rate
        t > T && return i
        cs ./= rate
        r = rand()
        j = findfirst(>(r), cs)
        i += ni[j]
    end
end

""" computes the vector of all rates to the cell `i` neighbours """
neighborrates(i::CartesianIndex, D, ni=neighboroffsets(D)) =
    map(i .+ ni) do j
        j in CartesianIndices(D) ? D[j] / D[i] : 0.0
    end

""" generate the relative cartesian indices pointing to all neighbours,
i.e. (1, 0, 0), (-1, 0, 0), (0, 1, 0), ..."""
neighboroffsets(x::Array) = neighboroffsets(ndims(x))
neighboroffsets(dim) =
    [CartesianIndex(setindex!(zeros(Int, dim), dir, i)...)
     for i in 1:dim for dir in (-1, 1)]

""" estimate K(chi) via Monte-Carlo using the Gillespie simulation """
Kchi_gillespie(start, chi, D, tau, nkoop) =
    1 / nkoop * sum(1:nkoop) do _
        chi[gillespie(start, D, tau), :]
    end

""" Compute the macroscopic rate approximation """
function pre(chi::Tensor, D::Tensor, tau, nstart, nkoop)
    @assert ndims(chi) == ndims(D) + 1

    starts = rand(CartesianIndices(D), nstart)
    x = stack([chi[s, :] for s in starts])'
    Kx = stack(starts) do s
        Kchi_gillespie(s, chi, D, tau, nkoop)
    end'

    # TODO: normalize?

    Kc = pinv(x) * Kx
    Qc = log(Kc ./ tau)
end


""" as part 1 in the article """
function example(; nx=50, tau=1.0, nkoop=100, nstart=10, nchi=2)
    V1(x) = (x^2 - 1)^2 + 1 * x
    V2(y) = 2 * y^2
    V12(x, y) = x * y
    Vc(x, c=1) = V1(x[1]) + V2(x[2]) + c * V12(x[1], x[2])

    grid = range(-3.4, 3.4, nx)

    potentials = [x -> V1(x[1]), x -> V2(x[1]), x -> V12(x[1], x[2])]
    indices = [[1], [2], [1, 2]]

    # TODO: adjust beta (and h?)
    beta = 1
    h = step(grid)

    Q1 = SqraCore.sqra_grid(V1.(grid); beta, h) |> collect  # collect to get dense
    Q2 = SqraCore.sqra_grid(V2.(grid); beta, h) |> collect

    # compute isolated membership functions
    chi1 = PCCAPlus.pcca(Q1, nchi)
    chi2 = PCCAPlus.pcca(Q2, nchi)

    # compute combined membership function
    chi = stack(c1 .* c2' for c1 in eachcol(chi1) for c2 in eachcol(chi2))

    # compute coupled stationary density
    D = compute_D(potentials, indices, [grid, grid])

    # PRE for coupled rate matrix with combined memberships as intial guess
    Q = pre(chi, D, tau, nstart, nkoop)

    # TODO: compare to coupled memberships obtained from from `D` using
    # tensor `eigenfuns` => PCCA+
    # sparse `sparse_Q` => PCCA+

    # TODO: extend to higher dims (n interacting "particle-potentials")
    # Q = [SqraCore.sqra_grid(V.(grid); beta=1, h=1) for V in potentials[1:ndims]]
    # chis = [PCCAPlus.pcca(Q[i], nchi) for i in 1:length(Q)]
    # outerprod(c...) = reshape(kron(reverse(c)), length.(c)...)
    # chi = stack(outerprod(c...) for c in Iterators.product(chis))

    # TODO: alternative: extend to higher dims (2 potentials + 1 interaction)
end