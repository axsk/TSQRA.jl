Tensor = AbstractArray

import ExponentialUtilities: expv


""" compute the kronecker product as tensor """
function tensorprod(c)
    reshape(kron(reverse(c)...), length.(c)...)
end

""" compute all possible tensor products between the columns of the provided matrices """
function alltensorprods(As...)
    mapreduce(hcat, Iterators.product(eachcol.(As)...)) do c
        tensorprod(c) |> vec
    end
end

### PRE 2

""" Computation of Qc via multistep (t->t+dt) PRE with exact propagation (expv) """
function pre2(Q, chi::AbstractMatrix, dt, t0=0, normalize=false)
    chi1 = expv(t0, Q, chi)
    chi2 = expv(dt, Q, chi1)
    normalize && rownormalize!.((chi1, chi2))
    Kc = pinv(chi1) * chi2
    Qc = log(Kc) ./ dt
    return Qc
end

rownormalize!(x) = x ./= sum(x, dims=2)

expv(t, A, b::AbstractMatrix) = stack(c -> expv(t, A, c), eachcol(b))

### PRE 1

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


""" estimate K(chi) via Monte-Carlo using the Gillespie simulation """
function Kchi_gillespie(start, chi, D, tau, nkoop, beta)
    1 / nkoop * sum(1:nkoop) do _
        chi[gillespie(start, D, tau, beta), :]
    end
end

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
