using LinearAlgebra
using Random: randexp

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

function neighborrates(i::CartesianIndex, Q, ni=neighboroffsets(Q))
    map(ni) do n
        j = i + n
        j in CartesianIndices(Q) ? Q[j] / Q[i] : 0.0
    end
end

neighboroffsets(x::Array) = neighboroffsets(ndims(x))
function neighboroffsets(dim=5)
    inds = CartesianIndex{dim}[]
    x = zeros(Int, dim)
    for i in 1:dim
        x .= 0
        x[i] = 1
        push!(inds, CartesianIndex(x...))
        x[i] = -1
        push!(inds, CartesianIndex(x...))
    end
    collect(inds)
end
