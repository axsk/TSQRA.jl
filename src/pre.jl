using LinearAlgebra: CartesianIndex, pinv
using Random: randexp

using SqraCore
using PCCAPlus
include("tsqra_simple.jl")

include("apply_a.jl")
include("tensorops.jl")
include("eigen.jl")

Tensor = AbstractArray

### PRE 1

""" Gillespie for SQRA via stationary tensor D """
function gillespie(i::CartesianIndex, D::Tensor, T::Float64, beta)
    ni = neighboroffsets(D)
    t = 0.0
    T = T * beta # rescale time to amount for Q = 1/beta * sqrt(p_j/p_i)
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
function neighborrates(i::CartesianIndex, D, ni=neighboroffsets(D))
    map(i .+ ni) do j
        j in CartesianIndices(D) ? D[j] / D[i] : 0.0
    end
end

""" generate the relative cartesian indices pointing to all neighbours,
i.e. (1, 0, 0), (-1, 0, 0), (0, 1, 0), ..."""
neighboroffsets(x::Array) = neighboroffsets(ndims(x))
function neighboroffsets(dim)
    [CartesianIndex(setindex!(zeros(Int, dim), dir, i)...)
     for i in 1:dim for dir in (-1, 1)]
end

""" estimate K(chi) via Monte-Carlo using the Gillespie simulation """
function Kchi_gillespie(start, chi, D, tau, nkoop, beta)
    1 / nkoop * sum(1:nkoop) do _
        chi[gillespie(start, D, tau, beta), :]
    end
end

""" Compute the macroscopic rate approximation """
function pre(chi::Tensor, D::Tensor, tau, nstart, nkoop, beta=getbeta())
    @assert ndims(chi) == ndims(D) + 1

    starts = rand(CartesianIndices(D), nstart)
    x = stack([chi[s, :] for s in starts])'
    Kx = stack(starts) do s
        Kchi_gillespie(s, chi, D, tau, nkoop, beta)
    end'
    Kx = Kx

    Kc = pinv(x) * Kx
    Kc ./= sum(Kc, dims=2) # rownormalize?

    Qc = log(Kc) ./ tau
    return Qc
end


### PRE 2
using ExponentialUtilities

rownormalize!(x) = x ./= sum(x, dims=2)
""" Computation of Qc via multistep (t->t+dt) PRE with exact propagation (expv) """
function pre2(Q, chi::AbstractMatrix, dt, t0=0, normalize=false)
    chi1 = expv(t0, Q, chi)
    chi2 = expv(dt, Q, chi1)
    normalize && rownormalize!.((chi1, chi2))
    Kc = pinv(chi1) * chi2
    Qc = log(Kc) ./ dt
    return Qc
end

import ExponentialUtilities: expv

# expend expv to colums of matrices
function expv(t, A, b::AbstractMatrix)
    stack(c -> expv(t, A, c), eachcol(b))
end

### EXAMPLES

""" as part 1 in the article """
function example1(; nx=50, dt=1.0, nkoop=100, nstart=10, nchi=2, t0=0)
    V1(x) = (x^2 - 1)^2 + 1 * x
    V2(y) = 2 * y^2
    V12(x, y) = x * y
    Vc(x, c=1) = V1(x[1]) + V2(x[2]) + c * V12(x[1], x[2])

    grid = range(-3.4, 3.4, nx)

    potentials = [x -> V1(x[1]), x -> V2(x[1]), x -> V12(x[1], x[2])]
    indices = [[1], [2], [1, 2]]

    # TODO: adjust beta (and h?)
    beta = getbeta()
    h = step(grid)

    @time Q1 = SqraCore.sqra_grid(V1.(grid); beta, h) |> collect  # collect to get dense
    Q2 = SqraCore.sqra_grid(V2.(grid); beta, h) |> collect

    # compute isolated membership functions
    @time chi1 = PCCAPlus.pcca(Q1, nchi)[1]
    chi2 = PCCAPlus.pcca(Q2, nchi)[1]

    # compute combined membership function
    chi = stack(c1 .* c2' for c1 in eachcol(chi1) for c2 in eachcol(chi2))

    # compute coupled stationary density
    @time D = compute_D(potentials, indices, [grid, grid])

    # PRE for coupled rate matrix with combined memberships as intial guess
    @time Qc1 = pre(chi, D, dt, nstart, nkoop)

    # PRE 2
    Q = QTensor(D)
    chif = reshape(chi, :, size(chi)[end])
    @time Qc2 = pre2(Q, chif, dt, t0)

    # PCCA on full Q
    Qs = sparse_Q(D; beta)
    @time chic = pcca(Qs, size(chif, 2), solver=KrylovSolver())[1]
    @time Qc = pinv(chic) * Qs * chic

    NamedTuple(Base.@locals)
end



function example2(; nx=5, dt=1.0, nkoop=100, nstart=10, nchi=2, nsys=3, t0=0.0, beta=getbeta())
    V(x) = (x[1]^2 - 1)^2
    Vc(x) = abs2(x[1] - x[2]) / 2

    grid1 = range(-2, 2, nx)  # corresponds to lucas Nedges = 6, a = 2.5
    grid = fill(grid1, nsys)

    potentials = [[V for i in 1:nsys]; [Vc for i in 1:(nsys-1)]]
    indices = [[[i] for i in 1:nsys]; [[i, i + 1] for i in 1:(nsys-1)]]

    h = step(grid1)

    Qi = SqraCore.sqra_grid(V.(grid1); beta, h) |> collect

    chi1 = PCCAPlus.pcca(Qi, nchi).chi

    allchis = outerprodprod((chi1 for i in 1:nsys)...)

    chi = allchis[:, [1]]
    chi = allchis
    # compute coupled stationary density
    D = compute_D(potentials, indices, grid, beta)
    Q = QTensor(D, beta)

    #Qc1 = pre(chi, D, dt, nstart, nkoop)


    Qc2 = pre2(Q, chi, dt, t0)

    if length(grid)^nsys < 1000
        #(; Qs, chic, Qc) = Qc_full(Q, nchi^nsys)
    end

    @exfiltrate
    NamedTuple(Base.@locals)
end

function outerprod(c)
    reshape(kron(reverse(c)...), length.(c)...)
end

function outerprodprod(As...)
    mapreduce(hcat, Iterators.product(eachcol.(As)...)) do c
        outerprod(c) |> vec
    end
end

function Qc_full(Q, nc)
    Qs = sparse(Q)
    # could use direct eigensolve + pcca
    chic = pcca(Qs, nc, solver=KrylovSolver(), optimize=true)[1]
    Qc = pinv(chic) * Qs * chic
    return (; Qs, chic, Qc)
end

function kroneigenfuns(chi, Q, dim, nc)
    Qc = pinv(chi) * Q * chi
    @show v = diag(Qc)
    @show cv = collect(zip(eachcol(chi), v))
    r = []
    for i in Iterators.product([cv for i in 1:dim]...)
        @show vecs, eigenvals = zip(i...)
        push!(r, (outerprod(vecs), sum(eigenvals)))
        #@show size(outerprod(c))
    end
    sort(r, by=x -> x[2])
end

using Plots
function plot_dt_dependence!(; dts=[0.1, 1, 2, 5, 10, 15] .* 0.1, kwargs...)

    (; D, nstart, nkoop, Q, chi, t0) = NamedTuple(kwargs)

    p1(dt) = pre(chi, D, dt, nstart, nkoop)
    p2(dt) = pre2(Q, chi, dt, t0)

    @time x = stack(dts) do dt
        diag(p2(dt))
    end .|> real
    p2 = plot(dts, x', title="PRE2"; legend=false)
end

# compute the coupled chi directly from the coupled Q
function chicoup()
    Dcoup = compute_D(potentials[1:3], indices[1:3], grid, getbeta())
    Qcoup = sparse_Q(Dcoup) ./ getbeta()
    chicoup = pcca(Qcoup, 8, solver=KrylovSolver(), optimize=true)[1]
end

function preluca(; Qs=Qs, Nd=nsys, chi=chicoup())
    KTAU = [0.1, 1, 2, 5, 10, 15] .* 0.1
    rates = zeros((2^Nd, 2^Nd, length(KTAU)))
    for (k, t) in enumerate(KTAU)
        tildeK = exp(collect(Qs) .* t)
        chi1 = tildeK * chi
        M = pinv(chi1) * chi
        hatQc = log(inv(M)) ./ t
        rates[:, :, k] = hatQc
    end

    plot()
    for i in 1:8
        plot!(KTAU, rates[i, i, :])
    end
    plot!() |> display
    return
end
