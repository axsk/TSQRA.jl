using LinearAlgebra: CartesianIndex, pinv
using Random: randexp
using PCCAPlus
using SqraCore

""" Gillespie for SQRA Tensor Q """
function gillespie(i::CartesianIndex, Q::AbstractArray, T::Float64)
    ni = neighboroffsets(Q)
    t = 0.0
    while true
        out = neighborrates(i, Q, ni)
        cs = cumsum(out)
        rate = cs[end]
        t += randexp() / rate
        t > T && return i
        cs ./= rate
        r = rand()
        j = findfirst(.>(r), cs)::Int
        i += ni[j]
        j += 1
    end
end

neighborrates(i::CartesianIndex, D, ni=neighboroffsets(D)) =
    map(ni) do n
        j = i + n
        j in CartesianIndices(D) ? D[j] / D[i] : 0.0
    end

neighboroffsets(x::Array) = neighboroffsets(ndims(x))
function neighboroffsets(dim)
    inds = CartesianIndex{dim}[]
    x = zeros(Int, dim)
    for i in 1:dim
        x .= 0
        x[i] = 1
        push!(inds, CartesianIndex(x...))
        x[i] = -1
        push!(inds, CartesianIndex(x...))
    end
    return inds
end

Kchi_gillespie(start, chi, D, tau, nkoop) =
    1 / nkoop * sum(1:nkoop) do _
        chi[gillespie(start, D, tau), :]
    end

function pre(chi, D, tau, nkoop, nstart)
    starts = rand(CartesianIndices(Qc), nstart)

    x = stack([chi[s, :] for s in starts])'
    Kx = stack(starts) do s
        Kchi_gillespie(s, chi, D, tau, nkoop)
    end'

    Kc = pinv(x) * Kx
    Qc = log(Kc ./ tau)
end

using SqraCore
using PCCAPlus
include("tsqra_simple.jl")

function example(; nx=50, tau=1.0, nkoop=100, nstart=10, nchi=2)
    V1(x) = (x^2 - 1)^2 + 1 * x
    V2(y) = 2 * y^2
    V12(x, y) = x * y
    Vc(x, c=1) = V1(x[1]) + V2(x[2]) + c * V12(x[1], x[2])

    grid = range(-3.4, 3.4, nx)

    potentials = [x -> V1(x[1]), x -> V2(x[1]), x -> V12(x[1], x[2])]
    indices = [[1], [2], [1, 2]]

    Q1 = SqraCore.sqra_grid(V1.(grid); beta=1) |> collect
    Q2 = SqraCore.sqra_grid(V2.(grid); beta=1) |> collect

    chi1 = PCCAPlus.pcca(Q1, nchi)
    chi2 = PCCAPlus.pcca(Q2, nchi)

    D = compute_D(potentials, indices, [grid, grid])

    chi = stack(c1 .* c2' for c1 in eachcol(chi1) for c2 in eachcol(chi2))

    Q = pre(chi, D, tau, nkoop, nstart)
end