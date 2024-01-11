using LinearAlgebra
using Strided

const kB = 0.008314463
beta(T=300) = 1 / kB / T
DiffusionConstant(T=300, mass=1, gamma=1) = kB * T / mass / gamma
flux(dx, D=DiffusionConstant()) = D / dx^2
flux(dx::AbstractRange, D) = flux(step(dx), D)

function tensor_sqra(v::Array; beta=beta(), clip=Inf)
    v = clippotential(v, clip)
    D = exp.((-beta / 2) .* v) #.* (D / delta^2)
    E = compute_E(D)
    return D, E
end

function clippotential(t, clip)
    t = replace(t, NaN => Inf)
    t[t.>clip] .= Inf
    return t
end

function compute_E(D)
    apply_Qo(ones(length(D)), vec(D), size(D))
end

# VALUEFIX
Di(D) = replace(D, 0 => convert(eltype(D), Inf))

apply_Q(x, D, E) = reshape(apply_Q(vec(x), vec(D), vec(E), size(D)), size(x))
apply_AE(x, E) = reshape(apply_AE(vec(x), vec(E), size(E)), size(x))

function apply_Q(x::AbstractVector, D::AbstractVector, E::AbstractVector, s)
    y = apply_Qo(x, D, s)
    y .-= E .* x
    return y
end

function apply_Qo(x::AbstractVector, D::AbstractVector, s)
    y = vec(apply_A_banded(x .* D, s)) ./ Di(D)
end

function apply_AE(x::AbstractVector, E::AbstractVector, s)
    y = vec(apply_A_banded(x, s))
    y .-= E .* x
    return y
end

""" modal_sum!(x,y,modes)

Widened hadamard sum.
Adds y to x along the dimensions specified in the array `modes`,
broadcasting over the other dimensions."""
function modal_sum!(x, y, modes)
    s = ones(Int, length(size(x)))
    s[modes] .= size(y)
    y = reshape(y, s...)
    x .+= y
    return x
end

using SparseArrays
function reconstruct_matrix_sparse(action, len)
    A = spzeros(len, len)
    for i in 1:len
        x = zeros(len)
        x[i] = 1
        A[:, i] = action(x)
    end
    return A
end

function sparse_Q(D, E=compute_E(D))
    Q(x) = apply_Q(x, vec(D), vec(E), size(D))
    reconstruct_matrix_sparse(Q, length(D))
end


# this supports eg. ExponentialUtilities.expv
struct QTensor{T}
    D::T
    E::T
end

QTensor(D) = QTensor(D, reshape(compute_E(D), size(D)))

Base.eltype(::QTensor{T}) where {T} = eltype(T)
Base.size(Q::QTensor, dim) = length(Q.D)
LinearAlgebra.ishermitian(Q::QTensor) = false
#LinearAlgebra.opnorm(A, p=Inf)

function LinearAlgebra.mul!(y, Q::QTensor, x)
    D, E = Q.D, Q.E
    D = reshape(D, :)
    E = reshape(E, :)
    #apply_A_banded!(y, x .* D, size(D)) ./ D
    #y .-= E .* x
    y .= apply_Q(x, D, E)
end

import Base.*
*(Q::QTensor, x) = mul!(similar(x), Q, x)

using SparseArrays
SparseArrays.sparse(Q::QTensor) = sparse_Q(Q.D, Q.E)